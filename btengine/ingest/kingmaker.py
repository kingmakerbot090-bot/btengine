"""Adapter: kingmaker data dump -> the engine's data contract.

The dump holds a raw Betfair exchange quote stream (odds.parquet) plus
match facts. Match facts come from the raw cricsheet JSON dump
(ipl_json/ or ipl_json.zip, parsed by btengine.ingest.cricsheet) when
present — the primary source — falling back to the pre-parsed
matches.parquet + ball_by_ball.parquet otherwise.

The quote stream mixes other markets into the same table (innings-runs
lines, top-batter, Yes/No props), so we keep only rows whose runner is
one of the match's two teams. Team names are normalised across sources
(the exchange kept "Royal Challengers Bangalore" after the 2024 rename).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from btengine.ingest.cricsheet import load_cricsheet

TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
}

_FACT_COLS = ["team1", "team2", "winner", "toss_winner",
              "toss_decision", "bats_first", "season"]


def _normalise(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_ALIASES)


def _facts_from_cricsheet(data_dir: Path) -> pd.DataFrame:
    src = data_dir / "ipl_json"
    if not src.is_dir():
        src = data_dir / "ipl_json.zip"
    return load_cricsheet(src)[["match_id"] + _FACT_COLS]


def _facts_from_parquet(data_dir: Path) -> pd.DataFrame:
    matches = pd.read_parquet(data_dir / "matches.parquet")
    bb = pd.read_parquet(data_dir / "ball_by_ball.parquet")
    batted_first = (
        bb[bb["innings"] == 1].groupby("match_id")["innings_team"].first()
    )
    matches["bats_first"] = matches["match_id"].map(batted_first)
    # fall back to the toss where ball-by-ball is missing
    toss_first = matches.apply(
        lambda r: r["toss_winner"] if r["toss_decision"] == "bat"
        else (r["team2"] if r["toss_winner"] == r["team1"] else r["team1"]),
        axis=1,
    )
    matches["bats_first"] = matches["bats_first"].fillna(toss_first)
    return matches[["match_id"] + _FACT_COLS]


def load_kingmaker_deliveries(data_dir, source: str = "auto") -> pd.DataFrame:
    """Ball-by-ball deliveries with team names normalised to match the
    odds stream (for btengine.align)."""
    from btengine.ingest.cricsheet import load_deliveries

    data_dir = Path(data_dir)
    if source == "auto":
        has_json = (data_dir / "ipl_json").is_dir() or \
            (data_dir / "ipl_json.zip").exists()
        source = "cricsheet" if has_json else "parquet"
    if source == "cricsheet":
        src = data_dir / "ipl_json"
        bb = load_deliveries(src if src.is_dir() else data_dir / "ipl_json.zip")
    else:
        bb = pd.read_parquet(data_dir / "ball_by_ball.parquet")
    bb["innings_team"] = _normalise(bb["innings_team"])
    return bb


def load_kingmaker(
    data_dir, source: str = "auto"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (odds, results, context) per btengine.ingest.odds contract.

    source: "cricsheet" (raw json dump), "parquet" (pre-parsed tables), or
    "auto" (cricsheet when present).
    """
    data_dir = Path(data_dir)
    if source == "auto":
        has_json = (data_dir / "ipl_json").is_dir() or \
            (data_dir / "ipl_json.zip").exists()
        source = "cricsheet" if has_json else "parquet"
    if source == "cricsheet":
        facts = _facts_from_cricsheet(data_dir)
    elif source == "parquet":
        facts = _facts_from_parquet(data_dir)
    else:
        raise ValueError(f"unknown source {source!r}")

    for col in ("team1", "team2", "winner", "toss_winner", "bats_first"):
        facts[col] = _normalise(facts[col])
    facts = facts.set_index("match_id")

    raw = pd.read_parquet(data_dir / "odds.parquet")
    raw = raw[raw["match_id"].notna()].copy()
    raw = raw[raw["market_status"] == "OPEN"]  # SUSPENDED/CLOSED aren't tradable
    raw["team"] = _normalise(raw["team"])

    # Keep only the two match-odds runners (drops props/lines/Tie runners).
    team1 = raw["match_id"].map(facts["team1"])
    team2 = raw["match_id"].map(facts["team2"])
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

    used = facts.loc[facts.index.isin(odds["match_id"].unique())]
    settled = used[used["winner"].notna()]  # drops abandoned / no-result

    results = settled.reset_index()[["match_id", "winner"]]
    context = settled.reset_index()[["match_id", "bats_first", "season"]]
    odds = odds[odds["match_id"].isin(results["match_id"])]
    return odds, results, context
