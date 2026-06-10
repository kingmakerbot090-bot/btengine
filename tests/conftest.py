import pandas as pd
import pytest

from btengine.synthetic import make_synthetic
from btengine.ticks import build_tick_table

T0 = pd.Timestamp("2025-04-01 14:00:00")


@pytest.fixture(scope="session")
def synth_ticks():
    odds, results, context = make_synthetic(n_matches=120, seed=11)
    return build_tick_table(odds, results, context)


@pytest.fixture()
def tiny_odds():
    """One hand-checkable match: fav A drifts from 1.50 to 2.25 (1.5x)."""
    rows = [
        # pre-off
        ("m1", T0 - pd.Timedelta(minutes=20), "A", 1.60, False),
        ("m1", T0 - pd.Timedelta(minutes=20), "B", 2.60, False),
        ("m1", T0 - pd.Timedelta(minutes=5), "A", 1.50, False),
        ("m1", T0 - pd.Timedelta(minutes=5), "B", 3.00, False),
        # in-play
        ("m1", T0, "A", 1.55, True),
        ("m1", T0, "B", 2.80, True),
        ("m1", T0 + pd.Timedelta(minutes=10), "A", 1.80, True),
        ("m1", T0 + pd.Timedelta(minutes=10), "B", 2.20, True),
        ("m1", T0 + pd.Timedelta(minutes=20), "A", 2.25, True),
        ("m1", T0 + pd.Timedelta(minutes=20), "B", 1.75, True),
        ("m1", T0 + pd.Timedelta(minutes=30), "A", 3.00, True),
        ("m1", T0 + pd.Timedelta(minutes=30), "B", 1.45, True),
    ]
    return pd.DataFrame(
        rows,
        columns=["match_id", "snapshot_time", "team", "decimal_odds", "inplay"],
    )


@pytest.fixture()
def tiny_results():
    return pd.DataFrame({"match_id": ["m1"], "winner": ["B"]})


@pytest.fixture()
def tiny_context():
    return pd.DataFrame(
        {"match_id": ["m1"], "bats_first": ["A"], "season": [2025]}
    )


@pytest.fixture()
def tiny_ticks(tiny_odds, tiny_results, tiny_context):
    return build_tick_table(tiny_odds, tiny_results, tiny_context)
