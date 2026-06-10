"""Raw-data loading and de-vigging.

Data contract (tidy odds table, one row per quote):
    match_id       str/int   market identifier
    snapshot_time  datetime  when the quote was observed
    team           str       runner name
    decimal_odds   float     best available decimal price
    inplay         bool      was the market in-play at this snapshot

Results table:  match_id, winner
Context table:  match_id, bats_first [, season]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ODDS_COLS = ["match_id", "snapshot_time", "team", "decimal_odds", "inplay"]

_BOOL_MAP = {
    True: True, False: False,
    "True": True, "False": False, "true": True, "false": False,
    1: True, 0: False, "1": True, "0": False,
}


def devig(odds_list) -> np.ndarray:
    """Strip the market overround: normalise implied probabilities to sum to 1.

    devig([1.5, 3.2]) -> proportional probabilities summing to exactly 1.
    """
    implied = 1.0 / np.asarray(odds_list, dtype=float)
    return implied / implied.sum()


def _coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.map(_BOOL_MAP).astype(bool)


def load_odds(path) -> pd.DataFrame:
    """Load and validate the tidy odds table from CSV."""
    odds = pd.read_csv(path)
    missing = [c for c in ODDS_COLS if c not in odds.columns]
    if missing:
        raise ValueError(f"odds table missing columns: {missing}")
    odds["snapshot_time"] = pd.to_datetime(odds["snapshot_time"])
    odds["inplay"] = _coerce_bool(odds["inplay"])
    odds["decimal_odds"] = odds["decimal_odds"].astype(float)
    if (odds["decimal_odds"] <= 1.0).any():
        raise ValueError("decimal_odds must be > 1.0")
    return odds[ODDS_COLS]


def load_results(path) -> pd.DataFrame:
    results = pd.read_csv(path)
    missing = [c for c in ("match_id", "winner") if c not in results.columns]
    if missing:
        raise ValueError(f"results table missing columns: {missing}")
    return results


def load_context(path) -> pd.DataFrame:
    context = pd.read_csv(path)
    if "match_id" not in context.columns:
        raise ValueError("context table missing match_id")
    return context
