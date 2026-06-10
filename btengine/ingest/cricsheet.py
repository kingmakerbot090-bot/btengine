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


def load_cricsheet(source) -> pd.DataFrame:
    """Parse a cricsheet dump (directory of *.json or a .zip) into one row
    per match. match_id is the cricsheet file stem."""
    source = Path(source)
    rows = []
    if source.is_dir():
        for f in sorted(source.glob("*.json")):
            rows.append(parse_match(json.loads(f.read_text()), f.stem))
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith(".json"):
                    rows.append(
                        parse_match(json.loads(zf.read(name)), Path(name).stem)
                    )
    else:
        raise ValueError(f"{source} is neither a directory nor a zip")
    if not rows:
        raise ValueError(f"no cricsheet json found in {source}")
    return pd.DataFrame(rows, columns=MATCH_COLS)
