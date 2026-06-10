"""Persistence: bet-by-bet logs for audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_betlog(betlog: pd.DataFrame, path) -> Path:
    """Write a betlog to CSV (creating parent dirs); returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    betlog.to_csv(path, index=False)
    return path
