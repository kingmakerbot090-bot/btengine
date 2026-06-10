"""Cricsheet ingest: parse the raw IPL JSON dump (cricsheet.org) into the
engine's results + context tables.

Source of truth for match facts, replacing trust in any pre-parsed
artifact: point it at the unzipped `ipl_json/` folder or the
`ipl_json.zip` straight from https://cricsheet.org/downloads/ipl_json.zip.

Settlement detail: a tied match decided by a super over carries
outcome = {"result": "tie", "eliminator": <team>}. Betfair MATCH_ODDS
settles on the super-over winner, so by default the eliminator counts as
the winner; abandoned/no-result matches keep winner = NaN and are dropped
downstream.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

MATCH_COLS = [
    "match_id", "date", "season", "team1", "team2",
    "toss_winner", "toss_decision", "winner", "result",
    "bats_first", "event_name",
]


def parse_match(data: dict, match_id: str) -> dict:
    info = data["info"]
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    result = "normal" if winner else outcome.get("result", "unknown")
    if winner is None:
        winner = outcome.get("eliminator")  # super-over settlement
    innings = data.get("innings", [])
    teams = info["teams"]
    return {
        "match_id": match_id,
        "date": info["dates"][0],
        "season": str(info.get("season", "")),
        "team1": teams[0],
        "team2": teams[1],
        "toss_winner": info["toss"]["winner"],
        "toss_decision": info["toss"]["decision"],
        "winner": winner,
        "result": result,
        "bats_first": innings[0]["team"] if innings else None,
        "event_name": (info.get("event") or {}).get("name"),
    }


def _iter_matches(source):
    source = Path(source)
    if source.is_dir():
        for f in sorted(source.glob("*.json")):
            yield f.stem, json.loads(f.read_text())
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith(".json"):
                    yield Path(name).stem, json.loads(zf.read(name))
    else:
        raise ValueError(f"{source} is neither a directory nor a zip")


def load_cricsheet(source) -> pd.DataFrame:
    """Parse a cricsheet dump (directory of *.json or a .zip) into one row
    per match. match_id is the cricsheet file stem."""
    rows = [parse_match(data, mid) for mid, data in _iter_matches(source)]
    if not rows:
        raise ValueError(f"no cricsheet json found in {source}")
    return pd.DataFrame(rows, columns=MATCH_COLS)


DELIVERY_COLS = [
    "match_id", "innings", "over", "ball", "innings_team",
    "runs_total", "legal_ball", "wicket",
]


def parse_deliveries(data: dict, match_id: str) -> list[dict]:
    rows = []
    for inn_no, inn in enumerate(data.get("innings", []), start=1):
        team = inn["team"]
        for over in inn.get("overs", []):
            for ball_no, dl in enumerate(over["deliveries"], start=1):
                extras = dl.get("extras", {})
                rows.append({
                    "match_id": match_id,
                    "innings": inn_no,
                    "over": over["over"],
                    "ball": ball_no,
                    "innings_team": team,
                    "runs_total": dl["runs"]["total"],
                    "legal_ball": not ("wides" in extras or "noballs" in extras),
                    "wicket": bool(dl.get("wickets")),
                })
    return rows


def load_deliveries(source) -> pd.DataFrame:
    """Parse a cricsheet dump into one row per delivery (the minimal
    schema btengine.align consumes; ball_by_ball.parquet matches it)."""
    rows = []
    for mid, data in _iter_matches(source):
        rows.extend(parse_deliveries(data, mid))
    if not rows:
        raise ValueError(f"no cricsheet json found in {source}")
    return pd.DataFrame(rows, columns=DELIVERY_COLS)
