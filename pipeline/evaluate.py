"""
pipeline/evaluate.py — Stage 4: Compare all models, pick best, save report
  - Reads metrics.json from train stage
  - Ranks by test RMSE
  - Copies best model to models/best_model.pkl
  - Saves evaluation report as JSON
"""

import sys
import json
import shutil
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger

log = get_logger("evaluate")


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def rank_models(metrics: dict) -> list:
    """Return models sorted by RMSE ascending (best first)."""
    ranked = sorted(
        [(name, m["rmse"], m.get("params", {}))
         for name, m in metrics.items()
         if m["rmse"] < 1e9],          # exclude failed models
        key=lambda x: x[1]
    )
    return ranked


def promote_best(best_name: str, models_dir: Path):
    """Copy best model file to best_model.pkl."""
    src = models_dir / f"{best_name}.pkl"
    dst = models_dir / "best_model.pkl"
    if src.exists():
        shutil.copy(src, dst)
        log.info(f"✓ Promoted {src.name} → best_model.pkl")
    else:
        log.warning(f"Model file not found: {src}")


if __name__ == "__main__":
    try:
        log.info("=== STAGE: evaluate ===")
        cfg = load_config()

        metrics_path = Path(cfg["models"]["metrics_path"])
        if not metrics_path.exists():
            log.error(f"Metrics not found at {metrics_path} — run train stage first")
            sys.exit(1)

        with open(metrics_path) as f:
            metrics = json.load(f)

        log.info(f"Loaded metrics for {len(metrics)} models")

        ranked = rank_models(metrics)

        log.info("\n" + "="*55)
        log.info("FINAL MODEL RANKING (lower RMSE = better)")
        log.info("="*55)
        for rank, (name, rmse, params) in enumerate(ranked, 1):
            marker = " ← BEST" if rank == 1 else ""
            log.info(f"  #{rank}  {name:<12}  RMSE = {rmse:.2f}{marker}")
        log.info("="*55)

        if not ranked:
            log.error("No valid models found — cannot pick best")
            sys.exit(1)

        best_name, best_rmse, best_params = ranked[0]
        log.info(f"Best model: {best_name}  RMSE={best_rmse:.2f}")

        # Promote best model
        models_dir = Path(cfg["models"]["output_dir"])
        promote_best(best_name, models_dir)

        # Save best model info (used by FastAPI)
        info = {
            "best_model":    best_name,
            "best_model_path": str(models_dir / "best_model.pkl"),
            "best_rmse":     best_rmse,
            "best_params":   best_params,
            "all_models":    [{"name": n, "rmse": r} for n, r, _ in ranked],
            "evaluated_at":  datetime.now().isoformat(),
        }
        info_path = Path(cfg["models"]["best_model_info"])
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        log.info(f"Best model info saved → {info_path}")

        # Save full report
        Path("reports").mkdir(exist_ok=True)
        report = {
            "summary":    info,
            "all_metrics": metrics,
            "ranking":    [{"rank": i+1, "model": n, "rmse": r}
                           for i, (n, r, _) in enumerate(ranked)],
        }
        report_path = "reports/evaluation.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Full evaluation report → {report_path}")
        log.info("Evaluate stage complete ✓")

    except Exception as e:
        log.error(f"Evaluate FAILED: {e}", exc_info=True)
        sys.exit(1)
