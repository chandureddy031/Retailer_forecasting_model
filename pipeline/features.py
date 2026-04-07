"""
pipeline/features.py — Stage 2: Feature engineering + scaling
  - Lag features (lag_1, lag_7, lag_14)
  - Rolling mean/std
  - Calendar features (dow, month, year)
  - StandardScaler on numerical cols
  - Saves features.csv + scaler.pkl
"""

import sys
import pickle
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger

log = get_logger("features")


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Adding calendar features...")
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon
    df["month"]       = df["date"].dt.month
    df["year"]        = df["date"].dt.year
    df["quarter"]     = df["date"].dt.quarter
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    log.info("✓ Calendar: day_of_week, month, year, quarter, is_weekend")
    return df


def add_lag_features(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    log.info(f"Adding lag features for lags: {lags}")
    df = df.sort_values(["store_id", "product_id", "date"])

    for lag in lags:
        col = f"lag_{lag}"
        df[col] = (
            df.groupby(["store_id", "product_id"])["sales"]
              .shift(lag)
        )
        log.debug(f"  Created {col} — nulls: {df[col].isna().sum()}")

    return df


def add_rolling_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    log.info(f"Adding rolling features for windows: {windows}")
    df = df.sort_values(["store_id", "product_id", "date"])

    grp = df.groupby(["store_id", "product_id"])["sales"]
    for w in windows:
        df[f"rolling_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
        )
        log.debug(f"  Created rolling_mean_{w}, rolling_std_{w}")

    return df


def scale_features(df: pd.DataFrame, num_cols: list, scaler_path: str):
    log.info(f"Scaling numerical columns: {num_cols}")

    # Only scale cols that exist
    cols = [c for c in num_cols if c in df.columns]
    log.debug(f"Columns available for scaling: {cols}")

    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols].fillna(0))

    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    log.info(f"✓ Scaler saved → {scaler_path}")
    return df, scaler


def build_num_cols(cfg: dict) -> list:
    lags    = [f"lag_{l}"              for l in cfg["features"]["lag_days"]]
    rolls   = [f"rolling_mean_{w}"     for w in cfg["features"]["rolling_windows"]]
    rolls  += [f"rolling_std_{w}"      for w in cfg["features"]["rolling_windows"]]
    others  = ["price", "day_of_week", "month", "year", "quarter"]
    return lags + rolls + others


if __name__ == "__main__":
    try:
        log.info("=== STAGE: features ===")
        cfg = load_config()

        df = pd.read_csv(cfg["data"]["interim_path"], parse_dates=["date"])
        log.info(f"Loaded {len(df)} rows for feature engineering")

        df = add_calendar_features(df)
        df = add_lag_features(df, cfg["features"]["lag_days"])
        df = add_rolling_features(df, cfg["features"]["rolling_windows"])

        # Drop rows with NaN from lags (early rows)
        before = len(df)
        df.dropna(inplace=True)
        log.info(f"Dropped {before - len(df)} rows with NaN (lag warm-up) → {len(df)} rows remain")

        num_cols = build_num_cols(cfg)
        df, _    = scale_features(df, num_cols, cfg["data"]["scaler_path"])

        out = cfg["data"]["processed_path"]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        log.info(f"Saved features → {out}  shape:{df.shape}")
        log.info("Features stage complete ✓")

    except Exception as e:
        log.error(f"Features FAILED: {e}", exc_info=True)
        sys.exit(1)
