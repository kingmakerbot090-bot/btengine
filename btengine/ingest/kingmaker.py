"""Adapter: kingmaker parquet dump -> the engine's data contract.

The dump holds a raw Betfair exchange quote stream (odds.parquet) plus
cricsheet-style match data (matches.parquet, ball_by_ball.parquet). The
quote stream mixes other markets into the same table (innings-runs lines,
top-batter, Yes/No props), so we keep only rows whose runner is one of the
match's two teams. Team names are normalised across sources (the exchange
kept "Royal Challengers Bangalore" after the 2024 rename).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
}


def _normalise(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_ALIASES)


def load_kingmaker(data_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (odds, results, context) per btengine.ingest.odds contract."""
    data_dir = Path(data_dir)
    raw = pd.read_parquet(data_dir / "odds.parquet")
    matches = pd.read_parquet(data_dir / "matches.parquet")
    bb = pd.read_parquet(data_dir / "ball_by_ball.parquet")

    raw = raw[raw["match_id"].notna()].copy()
    raw = raw[raw["market_status"] == "OPEN"]  # SUSPENDED/CLOSED aren't tradable
    raw["team"] = _normalise(raw["team"])

    for col in ("team1", "team2", "winner", "toss_winner"):
        matches[col] = _normalise(matches[col])
    matches = matches.set_index("match_id")

    # Keep only the two match-odds runners (drops props/lines/Tie runners).
    team1 = raw["match_id"].map(matches["team1"])
    team2 = raw["match_id"].map(matches["team2"])
    raw = raw[(raw["team"] == team1) | (raw["team"] == team2)]

    odds = pd.DataFrame({
        "match_id": raw["match_id"],
        "snapshot_time": pd.to_datetime(
            raw["snapshot_time"], utc=True, format="ISO8601"
        ),
        "team": raw["team"],
        "decimal_odds": raw["decimal_odds"].astype(float),
        "inplay": raw["inplay"].astype(bool),
    })
    # Exchange re-sends quotes; keep the latest row per (match, time, runner)
    # so devig sees at most one price per runner per snapshot.
    odds = (
        odds.sort_values("snapshot_time")
        .drop_duplicates(["match_id", "snapshot_time", "team"], keep="last")
        .reset_index(drop=True)
    )

    used = matches.loc[matches.index.isin(odds["match_id"].unique())]
    settled = used[used["winner"].notna()]  # drops 'no result' / 'tie'

    results = settled.reset_index()[["match_id", "winner"]]

    # Who batted first: first innings of the ball-by-ball feed (full
    # coverage here), falling back to the toss for any gap.
    bb["innings_team"] = _normalise(bb["innings_team"])
    batted_first = bb[bb["innings"] == 1].groupby("match_id")["innings_team"].first()
    toss_first = settled.apply(
        lambda r: r["toss_winner"] if r["toss_decision"] == "bat"
        else (r["team2"] if r["toss_winner"] == r["team1"] else r["team1"]),
        axis=1,
    )
    context = pd.DataFrame({
        "match_id": settled.index,
        "bats_first": batted_first.reindex(settled.index).fillna(toss_first).values,
        "season": settled["season"].values,
    })

    odds = odds[odds["match_id"].isin(results["match_id"])]
    return odds, results, context
