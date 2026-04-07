"""
demo_train.py — Full pipeline simulation using sklearn only
  Simulates all 7 model types with sklearn equivalents so you can
  demo the full pipeline without installing xgboost/statsmodels/prophet.

  Model mapping (for demo only):
    AR       → LinearRegression     (lag-only features)
    ARIMA    → Ridge                (lag + diff features)
    ARIMAX   → Ridge + exog         (lags + promotions/holidays)
    SARIMA   → ElasticNet           (seasonal features added)
    SARIMAX  → ElasticNet + exog    (seasonal + exog)
    Prophet  → GradientBoosting     (calendar + fourier features)
    XGBoost  → RandomForest + RandomizedSearch (full feature matrix)

  Produces: models/*.pkl + models/metrics.json  (same schema as train.py)
  In production → swap this with pipeline/train.py using real models.
"""

import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
from logger import get_logger

log = get_logger("demo_train")


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"  Saved → {path}")


def time_split(X, y, test_frac=0.2):
    cut = int(len(X) * (1 - test_frac))
    return X[:cut], X[cut:], y[:cut], y[cut:]


# ── Feature builders for each model type ─────────────────────────────────────

def make_lag_features(daily: pd.DataFrame, lags=[1, 7]) -> np.ndarray:
    df = daily.copy()
    for l in lags:
        df[f"lag_{l}"] = df["sales"].shift(l)
    df.dropna(inplace=True)
    cols = [f"lag_{l}" for l in lags]
    return df[cols].values, df["sales"].values


def make_diff_features(daily: pd.DataFrame) -> tuple:
    """Lag + 1st-difference (simulates ARIMA differencing)."""
    df = daily.copy()
    df["lag_1"]  = df["sales"].shift(1)
    df["lag_7"]  = df["sales"].shift(7)
    df["diff_1"] = df["sales"].diff(1)
    df.dropna(inplace=True)
    return df[["lag_1", "lag_7", "diff_1"]].values, df["sales"].values


def make_exog_features(daily: pd.DataFrame) -> tuple:
    """Lag + exogenous (promotion, holiday, price)."""
    df = daily.copy()
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df.dropna(inplace=True)
    cols = ["lag_1", "lag_7", "promotion", "is_holiday", "price"]
    return df[cols].values, df["sales"].values


def make_seasonal_features(daily: pd.DataFrame) -> tuple:
    """Lag + Fourier terms for seasonality (simulates SARIMA)."""
    df = daily.copy()
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    t = np.arange(len(df))
    for k in [1, 2]:
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * t / 7)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * t / 7)
    df.dropna(inplace=True)
    cols = ["lag_1", "lag_7", "sin_1", "cos_1", "sin_2", "cos_2"]
    return df[cols].values, df["sales"].values


def make_calendar_features(daily: pd.DataFrame) -> tuple:
    """Full calendar + Fourier (simulates Prophet)."""
    df = daily.copy()
    df["lag_7"]      = df["sales"].shift(7)
    df["dow"]        = df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    t = np.arange(len(df))
    for k in [1, 2, 3]:
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * t / 7)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * t / 7)
    df.dropna(inplace=True)
    cols = ["lag_7","dow","month","is_weekend","sin_1","cos_1","sin_2","cos_2","sin_3","cos_3"]
    return df[cols].values, df["sales"].values


# ── Model trainers ────────────────────────────────────────────────────────────

def train_ar(daily, test_frac, cfg):
    log.info("Training AR (LinearRegression on lags) ...")
    sc = cfg["models"]["ar"]["search"]
    best_r, best_mdl, best_lag = np.inf, None, None
    for lag in random.sample(sc["lags_range"], min(sc["n_samples"], len(sc["lags_range"]))):
        X, y = make_lag_features(daily, lags=list(range(1, lag+1)))
        Xtr, Xte, ytr, yte = time_split(X, y, test_frac)
        mdl = LinearRegression().fit(Xtr, ytr)
        r = rmse(yte, mdl.predict(Xte))
        log.debug(f"  AR lag={lag} RMSE={r:.2f}")
        if r < best_r:
            best_r, best_mdl, best_lag = r, mdl, lag
    log.info(f"  AR best_lag={best_lag} RMSE={best_r:.2f}")
    return best_mdl, best_r, {"lag": best_lag}


def train_arima(daily, test_frac, cfg):
    log.info("Training ARIMA (Ridge + diff features) ...")
    X, y = make_diff_features(daily)
    Xtr, Xte, ytr, yte = time_split(X, y, test_frac)
    sc = cfg["models"]["arima"]["search"]
    best_r, best_mdl, best_alpha = np.inf, None, None
    for _ in range(sc["n_samples"]):
        alpha = random.choice([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        mdl = Ridge(alpha=alpha).fit(Xtr, ytr)
        r = rmse(yte, mdl.predict(Xte))
        log.debug(f"  ARIMA alpha={alpha} RMSE={r:.2f}")
        if r < best_r:
            best_r, best_mdl, best_alpha = r, mdl, alpha
    log.info(f"  ARIMA best_alpha={best_alpha} RMSE={best_r:.2f}")
    return best_mdl, best_r, {"alpha": best_alpha, "order": "auto"}


def train_arimax(daily, test_frac, cfg):
    log.info("Training ARIMAX (Ridge + exog) ...")
    X, y = make_exog_features(daily)
    Xtr, Xte, ytr, yte = time_split(X, y, test_frac)
    sc = cfg["models"]["arimax"]["search"]
    best_r, best_mdl, best_alpha = np.inf, None, None
    for _ in range(sc["n_samples"]):
        alpha = random.choice([0.001, 0.01, 0.1, 1.0, 10.0])
        mdl = Ridge(alpha=alpha).fit(Xtr, ytr)
        r = rmse(yte, mdl.predict(Xte))
        log.debug(f"  ARIMAX alpha={alpha} RMSE={r:.2f}")
        if r < best_r:
            best_r, best_mdl, best_alpha = r, mdl, alpha
    log.info(f"  ARIMAX best_alpha={best_alpha} RMSE={best_r:.2f}")
    return best_mdl, best_r, {"alpha": best_alpha, "order": "auto", "exog": "promotion,holiday,price"}


def train_sarima(daily, test_frac, cfg):
    log.info("Training SARIMA (ElasticNet + Fourier seasonality) ...")
    X, y = make_seasonal_features(daily)
    Xtr, Xte, ytr, yte = time_split(X, y, test_frac)
    sc = cfg["models"]["sarima"]["search"]
    best_r, best_mdl, best_p = np.inf, None, None
    for _ in range(sc["n_samples"]):
        params = {
            "alpha":   random.choice([0.001, 0.01, 0.1, 1.0]),
            "l1_ratio": random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        }
        mdl = ElasticNet(**params, max_iter=2000).fit(Xtr, ytr)
        r = rmse(yte, mdl.predict(Xte))
        log.debug(f"  SARIMA params={params} RMSE={r:.2f}")
        if r < best_r:
            best_r, best_mdl, best_p = r, mdl, params
    log.info(f"  SARIMA best={best_p} RMSE={best_r:.2f}")
    return best_mdl, best_r, {**best_p, "seasonal": "Fourier s=7"}


def train_sarimax(daily, test_frac, cfg):
    log.info("Training SARIMAX (ElasticNet + Fourier + exog) ...")
    # Combine seasonal + exog features
    df = daily.copy()
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    t = np.arange(len(df))
    for k in [1, 2]:
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * t / 7)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * t / 7)
    df.dropna(inplace=True)
    cols = ["lag_1", "lag_7", "sin_1", "cos_1", "sin_2", "cos_2",
            "promotion", "is_holiday", "price"]
    X = df[cols].values
    y = df["sales"].values
    Xtr, Xte, ytr, yte = time_split(X, y, test_frac)
    sc = cfg["models"]["sarimax"]["search"]
    best_r, best_mdl, best_p = np.inf, None, None
    for _ in range(sc["n_samples"]):
        params = {
            "alpha":   random.choice([0.001, 0.01, 0.1, 1.0]),
            "l1_ratio": random.choice([0.1, 0.5, 0.9])
        }
        mdl = ElasticNet(**params, max_iter=2000).fit(Xtr, ytr)
        r = rmse(yte, mdl.predict(Xte))
        log.debug(f"  SARIMAX params={params} RMSE={r:.2f}")
        if r < best_r:
            best_r, best_mdl, best_p = r, mdl, params
    log.info(f"  SARIMAX best={best_p} RMSE={best_r:.2f}")
    return best_mdl, best_r, {**best_p, "seasonal+exog": True}


def train_prophet(daily, test_frac, cfg):
    log.info("Training Prophet-sim (GradientBoosting + calendar/Fourier) ...")
    X, y = make_calendar_features(daily)
    Xtr, Xte, ytr, yte = time_split(X, y, test_frac)
    sc = cfg["models"]["prophet"]
    # Randomized search over GBM hyperparams
    param_grid = {
        "n_estimators":    [100, 200, 300],
        "learning_rate":   [0.01, 0.05, 0.1],
        "max_depth":       [3, 4, 5],
        "subsample":       [0.7, 0.8, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=10, cv=tscv, scoring="neg_mean_squared_error",
        random_state=42, n_jobs=-1
    )
    search.fit(Xtr, ytr)
    best = search.best_estimator_
    r = rmse(yte, best.predict(Xte))
    log.info(f"  Prophet-sim best={search.best_params_} RMSE={r:.2f}")
    return best, r, {**search.best_params_, "changepoint_prior": sc["changepoint_prior_scale"]}


def train_xgboost(df_features, test_frac, cfg):
    log.info("Training XGBoost-sim (RandomForest + RandomizedSearch) ...")
    df = df_features.copy().sort_values("date")
    feat_cols = [c for c in df.columns if c not in
                 ["date", "sales", "store_id", "product_id"]]
    X = df[feat_cols].fillna(0).values
    y = df["sales"].values
    Xtr, Xte, ytr, yte = time_split(X, y, test_frac)
    sc = cfg["models"]["xgboost"]["search"]
    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "max_features":  ["sqrt", "log2", 0.7],
    }
    tscv = TimeSeriesSplit(n_splits=sc["cv"])
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=sc["n_iter"], cv=tscv,
        scoring=sc["scoring"], random_state=42, n_jobs=-1
    )
    search.fit(Xtr, ytr)
    best = search.best_estimator_
    r = rmse(yte, best.predict(Xte))
    log.info(f"  XGBoost-sim best={search.best_params_} RMSE={r:.2f}")
    return best, r, {k: str(v) for k, v in search.best_params_.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    random.seed(42)
    np.random.seed(42)

    log.info("=== STAGE: demo_train (7 models, sklearn-only) ===")
    cfg       = load_config()
    test_frac = cfg["data"]["test_split"]
    out_dir   = Path(cfg["models"]["output_dir"])
    out_dir.mkdir(exist_ok=True)

    # Load full feature set
    df = pd.read_csv(cfg["data"]["processed_path"], parse_dates=["date"])
    log.info(f"Loaded features: {df.shape}")

    # Aggregate daily for time-series models
    daily = (df.groupby("date")
               .agg(sales=("sales","sum"), promotion=("promotion","max"),
                    is_holiday=("is_holiday","max"), price=("price","mean"))
               .reset_index().sort_values("date"))
    log.info(f"Daily aggregated: {len(daily)} days")

    metrics = {}

    # Run all 7
    for name, fn, args in [
        ("ar",      train_ar,      (daily, test_frac, cfg)),
        ("arima",   train_arima,   (daily, test_frac, cfg)),
        ("arimax",  train_arimax,  (daily, test_frac, cfg)),
        ("sarima",  train_sarima,  (daily, test_frac, cfg)),
        ("sarimax", train_sarimax, (daily, test_frac, cfg)),
        ("prophet", train_prophet, (daily, test_frac, cfg)),
        ("xgboost", train_xgboost, (df,    test_frac, cfg)),
    ]:
        try:
            mdl, r, params = fn(*args)
            save_model(mdl, str(out_dir / f"{name}.pkl"))
            metrics[name] = {"rmse": round(r, 2), "params": params}
            log.info(f"✓ {name:<10} RMSE={r:.2f}")
        except Exception as e:
            log.error(f"✗ {name} FAILED: {e}", exc_info=True)
            metrics[name] = {"rmse": 9999.0, "params": {}}

    # Save metrics
    with open(cfg["models"]["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {cfg['models']['metrics_path']}")

    log.info("\n" + "="*55)
    log.info("FINAL MODEL RANKINGS")
    log.info("="*55)
    for name, m in sorted(metrics.items(), key=lambda x: x[1]["rmse"]):
        log.info(f"  {name:<12} RMSE = {m['rmse']:.2f}")
    log.info("="*55)
    log.info("Demo train complete ✓")
