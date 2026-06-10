"""Integration tests for the kingmaker parquet adapter.

Skipped unless the data dump is present (set KINGMAKER_DATA or place it
at data/kingmaker-data). The closing-line cross-check is the spec's
step-1 gate: our pipeline's pre-off reference prices must reproduce the
independently computed closing_lines.parquet exactly.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(os.environ.get("KINGMAKER_DATA", "data/kingmaker-data"))

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "odds.parquet").exists(),
    reason=f"kingmaker data not found at {DATA_DIR}",
)


@pytest.fixture(scope="module")
def kingmaker():
    from btengine.ingest.kingmaker import load_kingmaker

    return load_kingmaker(DATA_DIR)


@pytest.fixture(scope="module")
def real_ticks(kingmaker):
    from btengine.ticks import build_tick_table

    return build_tick_table(*kingmaker)


def test_contract_shape(kingmaker):
    odds, results, context = kingmaker
    assert list(odds.columns) == [
        "match_id", "snapshot_time", "team", "decimal_odds", "inplay"
    ]
    assert (odds["decimal_odds"] > 1.0).all()
    assert not odds.duplicated(["match_id", "snapshot_time", "team"]).any()
    assert set(results["match_id"]) == set(context["match_id"])
    assert results["winner"].notna().all()


def test_only_match_runners_survive(kingmaker):
    odds, results, _ = kingmaker
    # every quoted runner is one of the two teams that has a result row
    runners = odds.groupby("match_id")["team"].nunique()
    assert (runners == 2).all()


def test_closing_lines_match_independent_computation(real_ticks):
    cl = pd.read_parquet(DATA_DIR / "closing_lines.parquet").set_index("match_id")
    alias = {"Royal Challengers Bangalore": "Royal Challengers Bengaluru"}
    pre = real_ticks.groupby(["match_id", "team"])["preoff_price"].first()
    checked = 0
    for mid, row in cl.iterrows():
        theirs = {
            alias.get(row["team_a"], row["team_a"]): row["close_odds_a"],
            alias.get(row["team_b"], row["team_b"]): row["close_odds_b"],
        }
        for team, price in theirs.items():
            if (mid, team) in pre.index:
                assert pre.loc[(mid, team)] == pytest.approx(price, abs=1e-9)
                checked += 1
    assert checked > 500


def test_fav_tagging_is_sane(real_ticks):
    per_match = real_ticks.groupby(["match_id", "team"]).agg(
        fav=("is_fav", "first"), won=("won", "first")
    ).reset_index()
    favs = per_match[per_match["fav"]]
    # one fav per match, and favs win more than dogs at these odds ranges
    assert favs["match_id"].is_unique
    assert 0.45 < favs["won"].mean() < 0.70
