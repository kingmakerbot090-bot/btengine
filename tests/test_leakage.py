"""Integrity gate: entries must depend only on ticks <= the entry instant.

Two guards:
  1. corrupt every post-entry tick (odds, prob, drift) and assert the bet
     log's entries are byte-identical;
  2. an entry trigger that touches the settlement column must blow up —
     the result is structurally invisible at entry time.
"""

import numpy as np
import pandas as pd
import pytest

from btengine.engine import backtest
from btengine.strategy import Strategy, lay_drifting_fav

ENTRY_COLS = ["match_id", "team", "side", "entry_time", "entry_odds"]


def _corrupt_after_entries(ticks: pd.DataFrame, betlog: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    corrupted = ticks.copy()
    for _, bet in betlog.iterrows():
        mask = (corrupted["match_id"] == bet["match_id"]) & (
            corrupted["snapshot_time"] > bet["entry_time"]
        )
        n = int(mask.sum())
        assert n > 0, "fixture must have post-entry ticks to corrupt"
        corrupted.loc[mask, "odds"] = rng.uniform(1.01, 100.0, size=n)
        corrupted.loc[mask, "prob"] = rng.uniform(0.0, 1.0, size=n)
        corrupted.loc[mask, "drift_mult"] = (
            corrupted.loc[mask, "odds"] / corrupted.loc[mask, "preoff_price"]
        )
    return corrupted


@pytest.mark.parametrize("drift", [1.2, 1.5])
def test_corrupting_the_future_never_changes_an_entry(synth_ticks, drift):
    strategy = lay_drifting_fav(drift)
    betlog = backtest(strategy, synth_ticks)
    assert len(betlog) > 10, "fixture should produce a meaningful sample"

    corrupted = _corrupt_after_entries(synth_ticks, betlog)
    betlog2 = backtest(strategy, corrupted)

    pd.testing.assert_frame_equal(
        betlog[ENTRY_COLS].reset_index(drop=True),
        betlog2[ENTRY_COLS].reset_index(drop=True),
    )
    # settlement (and hence P&L) must also be untouched: results aren't ticks
    pd.testing.assert_series_equal(betlog["pnl"], betlog2["pnl"])


def test_result_column_is_invisible_to_entry_triggers(synth_ticks):
    cheat = Strategy(
        name="cheater",
        select="fav",
        side="lay",
        entry=lambda t: not t["won"],  # tries to read the settled result
        window=(0, 240),
    )
    with pytest.raises(KeyError):
        backtest(cheat, synth_ticks)
