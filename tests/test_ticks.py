import numpy as np
import pandas as pd
import pytest

from btengine.ingest.odds import devig
from btengine.ticks import build_tick_table, detect_off, preoff_reference
from tests.conftest import T0


def test_devig_sums_to_one_and_is_proportional():
    probs = devig([1.5, 3.2])
    assert probs.sum() == pytest.approx(1.0)
    # ratio of probs equals inverse ratio of odds
    assert probs[0] / probs[1] == pytest.approx(3.2 / 1.5)


def test_detect_off_is_first_inplay_quote(tiny_odds):
    assert detect_off(tiny_odds) == T0


def test_preoff_reference_takes_last_preoff_price(tiny_odds):
    ref = preoff_reference(tiny_odds)
    assert ref == {"A": 1.50, "B": 3.00}


def test_tick_table_columns_and_tags(tiny_ticks):
    a = tiny_ticks[tiny_ticks["team"] == "A"]
    b = tiny_ticks[tiny_ticks["team"] == "B"]
    # A is the pre-off fav, bats first, and lost
    assert a["is_fav"].all() and not b["is_fav"].any()
    assert a["bats_first"].all() and not b["bats_first"].any()
    assert not a["won"].any() and b["won"].all()
    assert (a["preoff_price"] == 1.50).all()
    assert (tiny_ticks["season"] == 2025).all()


def test_clock_is_minutes_since_off(tiny_ticks):
    a_inplay = tiny_ticks[(tiny_ticks["team"] == "A") & tiny_ticks["inplay"]]
    assert list(a_inplay["mins_since_off"]) == [0.0, 10.0, 20.0, 30.0]
    preoff = tiny_ticks[~tiny_ticks["inplay"]]
    assert (preoff["mins_since_off"] < 0).all()


def test_drift_mult_against_hand_calc(tiny_ticks):
    a20 = tiny_ticks[
        (tiny_ticks["team"] == "A") & (tiny_ticks["mins_since_off"] == 20.0)
    ].iloc[0]
    assert a20["drift_mult"] == pytest.approx(2.25 / 1.50)


def test_devigged_probs_sum_to_one_per_snapshot(tiny_ticks):
    sums = tiny_ticks.groupby(["match_id", "snapshot_time"])["prob"].sum()
    assert np.allclose(sums, 1.0)


def test_match_without_inplay_quotes_is_dropped(tiny_odds, tiny_results):
    preoff_only = tiny_odds[~tiny_odds["inplay"]].copy()
    with pytest.raises(ValueError):
        build_tick_table(preoff_only, tiny_results)
