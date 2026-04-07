"""
pipeline/train.py — Stage 3: Train all 7 models
  Models: AR, ARIMA, ARIMAX, SARIMA, SARIMAX, Prophet, XGBoost
  - Time-series models: trained on daily aggregated total sales
  - XGBoost: trained on full feature matrix
  - Randomized search for all models (p,d,q params OR sklearn grid)
  - Saves each model as .pkl + combined metrics.json
"""

import sys
import json
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from itertools import product as iterproduct

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from prophet import Prophet

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger("train")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"  Saved model → {path}")

def train_test_ts(series: pd.Series, test_frac: float):
    """Time-aware split for time series."""
    n    = len(series)
    cut  = int(n * (1 - test_frac))
    return series.iloc[:cut], series.iloc[cut:]


# ── Randomized search for SARIMAX-family ─────────────────────────────────────

def _random_sarimax_search(endog, exog=None, seasonal_order=None,
                            p_range=None, d_range=None, q_range=None,
                            P_range=None, D_range=None, Q_range=None,
                            s=7, n_samples=10, label="model"):
    """Sample random (p,d,q)(P,D,Q,s) combos; return best by AIC."""
    best_aic   = np.inf
    best_model = None
    best_order = None
    tried      = set()

    p_range = p_range or [0, 1, 2]
    d_range = d_range or [0, 1]
    q_range = q_range or [0, 1, 2]

    attempts = 0
    while len(tried) < n_samples and attempts < n_samples * 5:
        p = random.choice(p_range)
        d = random.choice(d_range)
        q = random.choice(q_range)
        order = (p, d, q)

        if seasonal_order is not None and P_range:
            P = random.choice(P_range)
            D = random.choice(D_range)
            Q = random.choice(Q_range)
            seas = (P, D, Q, s)
        else:
            seas = (0, 0, 0, 0)

        key = (order, seas)
        if key in tried:
            attempts += 1
            continue
        tried.add(key)
        attempts += 1

        try:
            mdl = SARIMAX(endog, exog=exog, order=order,
                          seasonal_order=seas,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False, maxiter=50)
            log.debug(f"  [{label}] order={order} seas={seas} AIC={mdl.aic:.1f}")
            if mdl.aic < best_aic:
                best_aic   = mdl.aic
                best_model = mdl
                best_order = (order, seas)
        except Exception as e:
            log.debug(f"  [{label}] order={order} failed: {e}")

    log.info(f"  [{label}] Best order={best_order}  AIC={best_aic:.1f}")
    return best_model, best_order, best_aic


# ── Individual model trainers ─────────────────────────────────────────────────

def train_ar(series, test, cfg):
    log.info("Training AR ...")
    sc  = cfg["models"]["ar"]["search"]
    best_rmse, best_model, best_lag = np.inf, None, None
    tried = set()
    samples = min(sc["n_samples"], len(sc["lags_range"]))
    for lag in random.sample(sc["lags_range"], samples):
        if lag in tried: continue
        tried.add(lag)
        try:
            mdl = AutoReg(series, lags=lag, old_names=False).fit()
            pred = mdl.predict(start=len(series), end=len(series)+len(test)-1)
            r = rmse(test.values, pred.values[:len(test)])
            log.debug(f"  AR lag={lag} RMSE={r:.2f}")
            if r < best_rmse:
                best_rmse, best_model, best_lag = r, mdl, lag
        except Exception as e:
            log.debug(f"  AR lag={lag} error: {e}")
    log.info(f"  AR best lag={best_lag} test_RMSE={best_rmse:.2f}")
    return best_model, best_rmse, {"lag": best_lag}


def train_arima(series, test, cfg):
    log.info("Training ARIMA ...")
    sc = cfg["models"]["arima"]["search"]
    mdl, order, aic = _random_sarimax_search(
        series, exog=None, seasonal_order=None,
        p_range=sc["p_range"], d_range=sc["d_range"], q_range=sc["q_range"],
        n_samples=sc["n_samples"], label="ARIMA"
    )
    if mdl is None:
        return None, np.inf, {}
    pred = mdl.forecast(steps=len(test))
    r = rmse(test.values, pred.values[:len(test)])
    log.info(f"  ARIMA test_RMSE={r:.2f}")
    return mdl, r, {"order": order}


def train_arimax(series, test, exog_train, exog_test, cfg):
    log.info("Training ARIMAX ...")
    sc = cfg["models"]["arimax"]["search"]
    mdl, order, aic = _random_sarimax_search(
        series, exog=exog_train, seasonal_order=None,
        p_range=sc["p_range"], d_range=sc["d_range"], q_range=sc["q_range"],
        n_samples=sc["n_samples"], label="ARIMAX"
    )
    if mdl is None:
        return None, np.inf, {}
    pred = mdl.forecast(steps=len(test), exog=exog_test)
    r = rmse(test.values, pred.values[:len(test)])
    log.info(f"  ARIMAX test_RMSE={r:.2f}")
    return mdl, r, {"order": order}


def train_sarima(series, test, cfg):
    log.info("Training SARIMA ...")
    sc = cfg["models"]["sarima"]["search"]
    mdl, order, aic = _random_sarimax_search(
        series, exog=None, seasonal_order=True,
        p_range=sc["p_range"], d_range=sc["d_range"], q_range=sc["q_range"],
        P_range=sc["P_range"], D_range=sc["D_range"], Q_range=sc["Q_range"],
        s=sc["s"], n_samples=sc["n_samples"], label="SARIMA"
    )
    if mdl is None:
        return None, np.inf, {}
    pred = mdl.forecast(steps=len(test))
    r = rmse(test.values, pred.values[:len(test)])
    log.info(f"  SARIMA test_RMSE={r:.2f}")
    return mdl, r, {"order": order}


def train_sarimax(series, test, exog_train, exog_test, cfg):
    log.info("Training SARIMAX ...")
    sc = cfg["models"]["sarimax"]["search"]
    mdl, order, aic = _random_sarimax_search(
        series, exog=exog_train, seasonal_order=True,
        p_range=sc["p_range"], d_range=sc["d_range"], q_range=sc["q_range"],
        P_range=sc["P_range"], D_range=sc["D_range"], Q_range=sc["Q_range"],
        s=sc["s"], n_samples=sc["n_samples"], label="SARIMAX"
    )
    if mdl is None:
        return None, np.inf, {}
    pred = mdl.forecast(steps=len(test), exog=exog_test)
    r = rmse(test.values, pred.values[:len(test)])
    log.info(f"  SARIMAX test_RMSE={r:.2f}")
    return mdl, r, {"order": order}


def train_prophet(daily_df, test_size, cfg):
    log.info("Training Prophet ...")
    pc  = cfg["models"]["prophet"]
    cut = len(daily_df) - test_size
    train_df = daily_df.iloc[:cut][["date", "sales", "promotion", "is_holiday"]].rename(
        columns={"date": "ds", "sales": "y"})
    test_df = daily_df.iloc[cut:][["date", "sales"]].rename(
        columns={"date": "ds", "sales": "y"})

    mdl = Prophet(
        yearly_seasonality=pc["yearly_seasonality"],
        weekly_seasonality=pc["weekly_seasonality"],
        daily_seasonality=pc["daily_seasonality"],
        changepoint_prior_scale=pc["changepoint_prior_scale"],
        seasonality_prior_scale=pc["seasonality_prior_scale"],
    )
    mdl.add_regressor("promotion")
    mdl.add_regressor("is_holiday")
    mdl.fit(train_df)

    future = mdl.make_future_dataframe(periods=len(test_df))
    future["promotion"]  = 0
    future["is_holiday"] = 0
    forecast = mdl.predict(future)
    pred = forecast.iloc[-len(test_df):]["yhat"].values
    r = rmse(test_df["y"].values, pred)
    log.info(f"  Prophet test_RMSE={r:.2f}")
    return mdl, r, {"changepoint_prior": pc["changepoint_prior_scale"]}


def train_xgboost(df, target_col, feature_cols, test_frac, cfg):
    log.info("Training XGBoost with RandomizedSearchCV ...")
    sc = cfg["models"]["xgboost"]["search"]

    df = df.copy().sort_values("date")
    cut = int(len(df) * (1 - test_frac))
    X   = df[feature_cols].fillna(0)
    y   = df[target_col]
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    tscv = TimeSeriesSplit(n_splits=sc["cv"])
    search = RandomizedSearchCV(
        xgb.XGBRegressor(tree_method="hist", random_state=42),
        param_distributions=sc["param_grid"],
        n_iter=sc["n_iter"],
        scoring=sc["scoring"],
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)
    r    = rmse(y_test.values, pred)
    log.info(f"  XGBoost best_params={search.best_params_}")
    log.info(f"  XGBoost test_RMSE={r:.2f}")
    return best, r, search.best_params_


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        log.info("=== STAGE: train (all 7 models) ===")
        cfg = load_config()

        df = pd.read_csv(cfg["data"]["processed_path"], parse_dates=["date"])
        log.info(f"Loaded features: {df.shape}")

        test_frac = cfg["data"]["test_split"]
        out_dir   = Path(cfg["models"]["output_dir"])
        out_dir.mkdir(exist_ok=True)

        # Aggregate daily for time-series models
        daily = (df.groupby("date")
                   .agg(sales=("sales","sum"), promotion=("promotion","max"),
                        is_holiday=("is_holiday","max"), price=("price","mean"))
                   .reset_index()
                   .sort_values("date"))
        log.info(f"Daily aggregated series: {len(daily)} days")

        n_test = int(len(daily) * test_frac)
        series_full  = daily.set_index("date")["sales"]
        exog_full    = daily.set_index("date")[["promotion", "is_holiday", "price"]]

        ts_train, ts_test       = train_test_ts(series_full, test_frac)
        exog_train, exog_test   = exog_full.iloc[:len(ts_train)], exog_full.iloc[len(ts_train):]

        metrics = {}

        # 1. AR
        mdl, r, params = train_ar(ts_train, ts_test, cfg)
        if mdl: save_model(mdl, str(out_dir / "ar.pkl"))
        metrics["ar"] = {"rmse": r, "params": params}

        # 2. ARIMA
        mdl, r, params = train_arima(ts_train, ts_test, cfg)
        if mdl: save_model(mdl, str(out_dir / "arima.pkl"))
        metrics["arima"] = {"rmse": r, "params": str(params)}

        # 3. ARIMAX
        mdl, r, params = train_arimax(ts_train, ts_test, exog_train, exog_test, cfg)
        if mdl: save_model(mdl, str(out_dir / "arimax.pkl"))
        metrics["arimax"] = {"rmse": r, "params": str(params)}

        # 4. SARIMA
        mdl, r, params = train_sarima(ts_train, ts_test, cfg)
        if mdl: save_model(mdl, str(out_dir / "sarima.pkl"))
        metrics["sarima"] = {"rmse": r, "params": str(params)}

        # 5. SARIMAX
        mdl, r, params = train_sarimax(ts_train, ts_test, exog_train, exog_test, cfg)
        if mdl: save_model(mdl, str(out_dir / "sarimax.pkl"))
        metrics["sarimax"] = {"rmse": r, "params": str(params)}

        # 6. Prophet
        mdl, r, params = train_prophet(daily, n_test, cfg)
        if mdl: save_model(mdl, str(out_dir / "prophet.pkl"))
        metrics["prophet"] = {"rmse": r, "params": params}

        # 7. XGBoost — feature cols
        feat_cols = [c for c in df.columns if c not in
                     ["date", "sales", "store_id", "product_id"]]
        mdl, r, params = train_xgboost(df, "sales", feat_cols, test_frac, cfg)
        save_model(mdl, str(out_dir / "xgboost.pkl"))
        metrics["xgboost"] = {"rmse": r, "params": {k: str(v) for k,v in params.items()}}

        # Save metrics
        with open(cfg["models"]["metrics_path"], "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Metrics saved → {cfg['models']['metrics_path']}")

        # Summary table
        log.info("\n" + "="*50)
        log.info("MODEL COMPARISON (test RMSE)")
        log.info("="*50)
        for name, m in sorted(metrics.items(), key=lambda x: x[1]["rmse"]):
            log.info(f"  {name:<12} RMSE = {m['rmse']:.2f}")
        log.info("="*50)
        log.info("Train stage complete ✓")

    except Exception as e:
        log.error(f"Train FAILED: {e}", exc_info=True)
        sys.exit(1)
