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


def test_cricsheet_parser_agrees_with_preparsed_tables():
    from btengine.ingest.cricsheet import load_cricsheet

    cs = load_cricsheet(DATA_DIR / "ipl_json").set_index("match_id")
    mt = pd.read_parquet(DATA_DIR / "matches.parquet").set_index("match_id")
    shared = cs.index.intersection(mt.index)
    assert len(shared) == len(mt)

    # winners agree wherever the pre-parsed table has one; cricsheet adds
    # super-over winners on top (tie -> eliminator)
    has_winner = mt.loc[shared, "winner"].notna()
    pd.testing.assert_series_equal(
        cs.loc[shared, "winner"][has_winner],
        mt.loc[shared, "winner"][has_winner],
        check_names=False,
    )
    ties = mt.loc[shared, "result"] == "tie"
    assert cs.loc[shared, "winner"][ties].notna().any()

    # bats-first agrees with the ball-by-ball first innings everywhere
    bb = pd.read_parquet(DATA_DIR / "ball_by_ball.parquet")
    first = bb[bb["innings"] == 1].groupby("match_id")["innings_team"].first()
    both = cs.index.intersection(first.index)
    pd.testing.assert_series_equal(
        cs.loc[both, "bats_first"], first.loc[both], check_names=False
    )


def test_source_parity_except_super_over_ties(kingmaker):
    from btengine.ingest.kingmaker import load_kingmaker

    odds_cs, results_cs, _ = kingmaker
    odds_pq, results_pq, _ = load_kingmaker(DATA_DIR, source="parquet")
    cs_ids = set(results_cs["match_id"])
    pq_ids = set(results_pq["match_id"])
    assert pq_ids <= cs_ids  # cricsheet settles super-over ties on top
    extra = cs_ids - pq_ids
    assert len(extra) <= 2
    pd.testing.assert_frame_equal(
        odds_cs[odds_cs["match_id"].isin(pq_ids)].reset_index(drop=True),
        odds_pq.reset_index(drop=True),
    )


def test_cricsheet_deliveries_match_preparsed_ball_by_ball():
    from btengine.ingest.cricsheet import load_deliveries

    bb = pd.read_parquet(DATA_DIR / "ball_by_ball.parquet")
    cs = load_deliveries(DATA_DIR / "ipl_json")
    sample = bb["match_id"].drop_duplicates().iloc[:25]
    cols = ["match_id", "innings", "over", "ball", "innings_team",
            "runs_total", "legal_ball", "wicket"]
    key = ["match_id", "innings", "over", "ball"]
    a = bb.loc[bb["match_id"].isin(sample), cols].sort_values(key)
    b = cs.loc[cs["match_id"].isin(sample), cols].sort_values(key)
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True),
        check_dtype=False,
    )


def test_alignment_wicket_jump_lift(real_ticks):
    from btengine.align import wicket_jump_lift

    bb = pd.read_parquet(DATA_DIR / "ball_by_ball.parquet")
    bb["innings_team"] = bb["innings_team"].replace(
        {"Royal Challengers Bangalore": "Royal Challengers Bengaluru"}
    )
    lift, n = wicket_jump_lift(real_ticks, bb)
    assert n > 3000
    assert lift > 1.5, f"alignment lost its signal: lift={lift:.2f}"


def test_attached_state_covers_the_stream(real_ticks):
    from btengine.align import attach_state

    bb = pd.read_parquet(DATA_DIR / "ball_by_ball.parquet")
    bb["innings_team"] = bb["innings_team"].replace(
        {"Royal Challengers Bangalore": "Royal Challengers Bengaluru"}
    )
    aligned = attach_state(real_ticks, bb)
    assert len(aligned) == len(real_ticks)
    inplay = aligned[aligned["inplay"]]
    assert inplay["est_innings"].notna().mean() > 0.95
    # both innings observed in nearly every match
    reach2 = inplay.groupby("match_id")["est_innings"].max()
    assert (reach2 >= 2).mean() > 0.9
    # the batting team is always one of the two runners
    known = inplay[inplay["batting_team"].notna()]
    assert known.groupby("match_id").apply(
        lambda g: g["batting_team"].isin(g["team"]).all()
    ).all()


def test_fav_tagging_is_sane(real_ticks):
    per_match = real_ticks.groupby(["match_id", "team"]).agg(
        fav=("is_fav", "first"), won=("won", "first")
    ).reset_index()
    favs = per_match[per_match["fav"]]
    # one fav per match, and favs win more than dogs at these odds ranges
    assert favs["match_id"].is_unique
    assert 0.45 < favs["won"].mean() < 0.70
