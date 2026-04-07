"""
dataset.py — Synthetic retail sales generator
  Produces 5000 rows: stores × products × dates
  Includes: trend, weekly seasonality, promotions, holidays, noise
"""

import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from logger import get_logger

log = get_logger("dataset")


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_sales(config: dict) -> pd.DataFrame:
    cfg = config["data"]
    np.random.seed(42)

    n_stores   = cfg["n_stores"]
    n_products = cfg["n_products"]
    n_rows     = cfg["n_rows"]

    # Dates: enough to hit ~n_rows after cross-join
    n_days = n_rows // (n_stores * n_products)
    dates  = pd.date_range(cfg["date_start"], periods=n_days, freq="D")

    log.info(f"Generating {n_stores} stores × {n_products} products × {n_days} days = {n_stores*n_products*n_days} rows")

    rows = []
    for store in range(1, n_stores + 1):
        store_base = np.random.uniform(50, 150)          # store-level demand offset
        for prod in range(1, n_products + 1):
            prod_base  = np.random.uniform(0.5, 2.0)     # product multiplier
            price_base = np.random.uniform(10, 200)       # base price

            for i, date in enumerate(dates):
                # Trend (slight upward)
                trend = i * 0.05

                # Weekly seasonality (peak Fri-Sat)
                weekly = 10 * np.sin(2 * np.pi * date.dayofweek / 7)

                # Monthly seasonality (peak mid-month)
                monthly = 5 * np.sin(2 * np.pi * date.day / 30)

                # Holiday (random ~8% of days)
                is_holiday = int(np.random.random() < 0.08)
                holiday_boost = 30 * is_holiday

                # Promotion (random ~15% of days)
                promotion = int(np.random.random() < 0.15)
                promo_boost = 20 * promotion

                # Price (varies slightly)
                price = round(price_base * np.random.uniform(0.9, 1.1), 2)

                # Final sales — floor at 0
                noise = np.random.normal(0, 8)
                sales = max(0, round(
                    (store_base + trend + weekly + monthly + holiday_boost + promo_boost)
                    * prod_base + noise
                ))

                rows.append({
                    "date":       date,
                    "store_id":   store,
                    "product_id": prod,
                    "sales":      sales,
                    "price":      price,
                    "promotion":  promotion,
                    "is_holiday": is_holiday,
                })

    df = pd.DataFrame(rows)
    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Sales stats — min:{df.sales.min()} max:{df.sales.max()} mean:{df.sales.mean():.1f}")
    log.debug(f"Date range: {df.date.min()} → {df.date.max()}")
    return df


def save(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Saved {len(df)} rows → {path}")


if __name__ == "__main__":
    try:
        log.info("=== STAGE: generate dataset ===")
        cfg = load_config()
        df  = generate_sales(cfg)
        save(df, cfg["data"]["raw_path"])
        log.info("Dataset generation complete ✓")
    except Exception as e:
        log.error(f"Dataset generation FAILED: {e}", exc_info=True)
        sys.exit(1)
