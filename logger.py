"""
logger.py — Session logger
  - One log file per calendar day (all stages share it)
  - Logs INFO/ERROR/DEBUG to file + console
  - Every module calls: from logger import get_logger; log = get_logger(__name__)
"""

import logging
import os
from datetime import datetime
from pathlib import Path


_FMT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Safe to call multiple times — handlers added once."""
    Path("logs").mkdir(exist_ok=True)

    log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times in same process
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # File handler — DEBUG and above
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler — INFO and above
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logger '{name}' initialised → {log_file}")
    return logger
