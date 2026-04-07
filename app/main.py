"""
app/main.py — FastAPI serving forecast + inventory recommendations
  Routes:
    GET  /               → dashboard (Jinja2)
    POST /predict        → run forecast
    GET  /models         → list trained models + metrics
    GET  /health         → health check
"""

import json
import pickle
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger

log = get_logger("app")

# ── Bootstrap ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
app  = Jinja2Templates(directory=str(ROOT / "app" / "templates"))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

INV = CFG["inventory"]


def load_model_info() -> dict:
    p = ROOT / CFG["models"]["best_model_info"]
    if p.exists():
        return json.loads(p.read_text())
    return {}


def load_best_model():
    p = ROOT / "models" / "best_model.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def load_raw_daily() -> pd.DataFrame:
    """Aggregate daily sales from processed features for context."""
    fp = ROOT / CFG["data"]["processed_path"]
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, parse_dates=["date"])
    return (df.groupby("date")
              .agg(sales=("sales", "sum"), promotion=("promotion", "max"),
                   is_holiday=("is_holiday", "max"), price=("price", "mean"))
              .reset_index()
              .sort_values("date"))


MODEL_INFO  = load_model_info()
BEST_MODEL  = load_best_model()
DAILY_DF    = load_raw_daily()


# ── Inventory logic ───────────────────────────────────────────────────────────

def inventory_decision(forecast: list, current_stock: float, cfg: dict) -> dict:
    """Core inventory decision engine."""
    pred = np.array(forecast)
    avg_demand  = float(np.mean(pred))
    max_demand  = float(np.max(pred))
    total_need  = float(np.sum(pred))
    lead_days   = cfg["lead_time_days"]
    safety      = (max_demand - avg_demand) * lead_days * cfg["safety_stock_factor"]
    required    = total_need + safety
    order_qty   = max(0.0, required - current_stock)

    # Alerts
    days_of_stock = (current_stock / avg_demand) if avg_demand > 0 else 999
    alerts = []
    if days_of_stock < cfg["stockout_threshold_days"]:
        alerts.append({"type": "danger", "msg": f"⚠️ Stockout risk in {days_of_stock:.1f} days!"})
    if current_stock > avg_demand * cfg["overstock_multiplier"] * len(forecast):
        alerts.append({"type": "warning", "msg": "📦 Overstock risk — consider pausing orders"})
    if not alerts:
        alerts.append({"type": "success", "msg": "✅ Stock levels healthy"})

    return {
        "avg_daily_demand":  round(avg_demand, 1),
        "total_forecast":    round(total_need, 1),
        "safety_stock":      round(safety, 1),
        "required_stock":    round(required, 1),
        "current_stock":     round(current_stock, 1),
        "order_quantity":    round(order_qty, 1),
        "days_of_stock":     round(days_of_stock, 1),
        "alerts":            alerts,
    }


# ── Forecasting ───────────────────────────────────────────────────────────────

def make_forecast(model, model_type: str, horizon: int) -> list:
    """Route forecast to correct model API."""
    if DAILY_DF.empty:
        return [100.0] * horizon          # fallback demo values

    series = DAILY_DF.set_index("date")["sales"]
    exog   = DAILY_DF.set_index("date")[["promotion", "is_holiday", "price"]]

    # Future exog (assume no promo/holiday, avg price)
    future_exog = pd.DataFrame(
        {"promotion": 0, "is_holiday": 0, "price": exog["price"].mean()},
        index=range(horizon)
    )

    try:
        if model_type == "prophet":
            future = model.make_future_dataframe(periods=horizon)
            future["promotion"]  = 0
            future["is_holiday"] = 0
            fc = model.predict(future)
            vals = fc.iloc[-horizon:]["yhat"].clip(lower=0).tolist()

        elif model_type == "xgboost":
            # Build simple future feature rows from last known values
            last_row = DAILY_DF.iloc[-1]
            rows = []
            for i in range(horizon):
                rows.append({
                    "day_of_week":    (last_row["date"] + timedelta(days=i+1)).dayofweek,
                    "month":          (last_row["date"] + timedelta(days=i+1)).month,
                    "year":           (last_row["date"] + timedelta(days=i+1)).year,
                    "quarter":        (last_row["date"] + timedelta(days=i+1)).quarter,
                    "is_weekend":     int((last_row["date"] + timedelta(days=i+1)).dayofweek >= 5),
                    "promotion":      0,
                    "is_holiday":     0,
                    "price":          float(last_row["price"]),
                    "lag_1":          float(series.iloc[-1]) if i == 0 else rows[-1].get("lag_1", series.mean()),
                    "lag_7":          float(series.iloc[-7]) if len(series) >= 7 else series.mean(),
                    "lag_14":         float(series.iloc[-14]) if len(series) >= 14 else series.mean(),
                    "rolling_mean_7": float(series.iloc[-7:].mean()),
                    "rolling_std_7":  float(series.iloc[-7:].std()),
                    "rolling_mean_14": float(series.iloc[-14:].mean()),
                    "rolling_std_14": float(series.iloc[-14:].std()),
                })
            feat_cols = [c for c in model.feature_names_in_ if c in rows[0]]
            X = pd.DataFrame(rows)[feat_cols].fillna(0)
            vals = model.predict(X).clip(min=0).tolist()

        else:  # SARIMAX-family (AR, ARIMA, ARIMAX, SARIMA, SARIMAX)
            if model_type in ("arimax", "sarimax"):
                vals = model.forecast(steps=horizon, exog=future_exog).clip(lower=0).tolist()
            else:
                vals = model.forecast(steps=horizon).clip(lower=0).tolist()

    except Exception as e:
        log.error(f"Forecast error [{model_type}]: {e}", exc_info=True)
        vals = [float(series.mean())] * horizon   # safe fallback

    return [round(v, 1) for v in vals]


# ── Routes ────────────────────────────────────────────────────────────────────

api = FastAPI(title="Retail Demand Forecasting API")


@api.get("/health")
def health():
    return {"status": "ok", "model_loaded": BEST_MODEL is not None,
            "best_model": MODEL_INFO.get("best_model", "none")}


@api.get("/models")
def list_models():
    return MODEL_INFO.get("all_models", [])


@api.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    ctx = {
        "request":      request,
        "model_info":   MODEL_INFO,
        "all_models":   MODEL_INFO.get("all_models", []),
        "best_model":   MODEL_INFO.get("best_model", "N/A"),
        "best_rmse":    round(MODEL_INFO.get("best_rmse", 0), 2),
        "model_ready":  BEST_MODEL is not None,
        "result":       None,
    }
    return app.TemplateResponse("index.html", ctx)


@api.post("/", response_class=HTMLResponse)
async def predict_form(
    request:       Request,
    horizon:       int   = Form(7),
    current_stock: float = Form(200),
):
    model_type = MODEL_INFO.get("best_model", "xgboost")
    forecast   = make_forecast(BEST_MODEL, model_type, horizon)

    future_dates = [
        (datetime.today() + timedelta(days=i+1)).strftime("%b %d")
        for i in range(horizon)
    ]

    inv = inventory_decision(forecast, current_stock, INV)

    log.info(f"Forecast horizon={horizon} stock={current_stock} order={inv['order_quantity']}")

    ctx = {
        "request":       request,
        "model_info":    MODEL_INFO,
        "all_models":    MODEL_INFO.get("all_models", []),
        "best_model":    MODEL_INFO.get("best_model", "N/A"),
        "best_rmse":     round(MODEL_INFO.get("best_rmse", 0), 2),
        "model_ready":   BEST_MODEL is not None,
        "result": {
            "forecast":      forecast,
            "future_dates":  future_dates,
            "horizon":       horizon,
            "current_stock": current_stock,
            "inventory":     inv,
        },
    }
    return app.TemplateResponse("index.html", ctx)
