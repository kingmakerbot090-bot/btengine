"""Price-path layer: turn raw quotes into the point-in-time tick table.

The tick table is the single source of truth downstream. Every column is
either live-knowable at its snapshot (odds, prob, mins_since_off,
preoff_price, drift_mult, is_fav, bats_first) or a settlement label (won)
which the execution engine hides from entry triggers.

The only in-match clock is minutes since the "off" (first in-play quote) —
never fraction-of-total-match, which would leak the match length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from btengine.ingest.odds import load_odds, load_results, load_context

# Settlement-only columns, structurally hidden from entry triggers (engine.py).
RESULT_COLS = ("won",)

TICK_COLS = [
    "match_id", "snapshot_time", "team", "odds", "prob", "inplay",
    "mins_since_off", "preoff_price", "drift_mult",
    "is_fav", "bats_first", "season", "won",
]


def detect_off(match_quotes: pd.DataFrame) -> pd.Timestamp:
    """The match clock origin: time of the first in-play quote (NaT if none)."""
    inplay = match_quotes.loc[match_quotes["inplay"], "snapshot_time"]
    return inplay.min() if len(inplay) else pd.NaT


def preoff_reference(match_quotes: pd.DataFrame, t0=None) -> dict:
    """Last pre-off price per runner: {team: decimal_odds}."""
    if t0 is None:
        t0 = detect_off(match_quotes)
    pre = match_quotes[~match_quotes["inplay"]]
    if not pd.isna(t0):
        pre = pre[pre["snapshot_time"] < t0]
    if pre.empty:
        return {}
    last = (
        pre.sort_values("snapshot_time")
        .groupby("team")["decimal_odds"]
        .last()
    )
    return last.to_dict()


def build_tick_table(
    odds: pd.DataFrame,
    results: pd.DataFrame,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the off-anchored, de-vigged tick table.

    Drops matches with no result, no in-play quotes, or fewer than two
    runners priced pre-off (no fav/dog tagging possible).
    """
    odds = odds.copy()
    odds["snapshot_time"] = pd.to_datetime(odds["snapshot_time"])
    winner_by_match = results.set_index("match_id")["winner"]
    ctx = context.set_index("match_id") if context is not None else None

    frames = []
    for match_id, g in odds.groupby("match_id", sort=False):
        if match_id not in winner_by_match.index:
            continue
        g = g.sort_values("snapshot_time").copy()
        t0 = detect_off(g)
        if pd.isna(t0):
            continue
        ref = preoff_reference(g, t0)
        if len(ref) < 2:
            continue

        g = g.rename(columns={"decimal_odds": "odds"})
        g["mins_since_off"] = (g["snapshot_time"] - t0).dt.total_seconds() / 60.0
        g["preoff_price"] = g["team"].map(ref)
        g["drift_mult"] = g["odds"] / g["preoff_price"]
        fav = min(ref, key=ref.get)
        g["is_fav"] = g["team"] == fav
        g["won"] = g["team"] == winner_by_match.loc[match_id]

        g["bats_first"] = False
        g["season"] = g["snapshot_time"].dt.year
        if ctx is not None and match_id in ctx.index:
            row = ctx.loc[match_id]
            if "bats_first" in ctx.columns:
                g["bats_first"] = g["team"] == row["bats_first"]
            if "season" in ctx.columns:
                g["season"] = row["season"]
        frames.append(g)

    if not frames:
        raise ValueError("no usable matches: check results join and inplay flags")
    ticks = pd.concat(frames, ignore_index=True)

    # De-vig per (match, snapshot): only where the full 2-runner book is quoted.
    implied = 1.0 / ticks["odds"]
    grp = ticks.groupby(["match_id", "snapshot_time"])["odds"]
    book = grp.transform(lambda o: (1.0 / o).sum())
    n_quoted = grp.transform("size")
    ticks["prob"] = np.where(n_quoted == 2, implied / book, implied)

    return ticks[TICK_COLS].sort_values(
        ["match_id", "snapshot_time", "team"]
    ).reset_index(drop=True)


def load_tick_table(
    odds_path,
    results_path,
    context_path=None,
    cache_path="data/ticks.pkl",
    rebuild: bool = False,
) -> pd.DataFrame:
    """Cached build: read the pickle cache unless rebuild=True or missing."""
    cache = Path(cache_path)
    if cache.exists() and not rebuild:
        return pd.read_pickle(cache)
    odds = load_odds(odds_path)
    results = load_results(results_path)
    context = load_context(context_path) if context_path else None
    ticks = build_tick_table(odds, results, context)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ticks.to_pickle(cache)
    return ticks
