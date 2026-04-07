"""
pipeline/ingest.py — Stage 1: Load & validate raw data
  - Checks nulls, duplicates, data types
  - Logs shape, dtypes, basic stats
  - Saves clean validated copy
"""

import sys
import pandas as pd
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger

log = get_logger("ingest")


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    log.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    log.info(f"Loaded shape: {df.shape}")
    log.debug(f"Columns: {list(df.columns)}")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Running validation checks...")

    # Null check
    nulls = df.isnull().sum()
    if nulls.any():
        log.warning(f"Nulls found:\n{nulls[nulls > 0]}")
        before = len(df)
        df = df.dropna()
        log.info(f"Dropped {before - len(df)} null rows")
    else:
        log.info("✓ No null values")

    # Duplicate check
    dupes = df.duplicated(subset=["date", "store_id", "product_id"]).sum()
    if dupes:
        log.warning(f"Found {dupes} duplicate rows — dropping")
        df = df.drop_duplicates(subset=["date", "store_id", "product_id"])
    else:
        log.info("✓ No duplicate rows")

    # Negative sales check
    neg = (df["sales"] < 0).sum()
    if neg:
        log.warning(f"Found {neg} negative sales — clipping to 0")
        df["sales"] = df["sales"].clip(lower=0)
    else:
        log.info("✓ No negative sales")

    # Type enforcement
    df["date"]       = pd.to_datetime(df["date"])
    df["store_id"]   = df["store_id"].astype(int)
    df["product_id"] = df["product_id"].astype(int)
    df["sales"]      = df["sales"].astype(float)
    log.info("✓ Data types enforced")

    # Log summary stats
    log.info(f"Final shape after validation: {df.shape}")
    log.info(f"Date range: {df.date.min().date()} → {df.date.max().date()}")
    log.info(f"Stores: {sorted(df.store_id.unique())}")
    log.info(f"Products: {sorted(df.product_id.unique())}")
    log.info(f"Sales — mean:{df.sales.mean():.1f} std:{df.sales.std():.1f} max:{df.sales.max()}")

    return df


def save(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Saved validated data → {path}")


if __name__ == "__main__":
    try:
        log.info("=== STAGE: ingest ===")
        cfg = load_config()
        df  = load_data(cfg["data"]["raw_path"])
        df  = validate(df)
        save(df, cfg["data"]["interim_path"])
        log.info("Ingest stage complete ✓")
    except Exception as e:
        log.error(f"Ingest FAILED: {e}", exc_info=True)
        sys.exit(1)
