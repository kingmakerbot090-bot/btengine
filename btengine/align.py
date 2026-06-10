"""Align the odds stream with ball-by-ball data — without per-ball timestamps.

Cricsheet deliveries carry no clock, and the odds stream is a 60-second
polled snapshot feed (the market stays OPEN through the innings break, so
there is no gap to anchor on). The bridge is a clock model built only from
live-knowable quantities:

    est_time(delivery k of innings 1) = off + k * seconds_per_delivery
    est_time(delivery k of innings 2) = off + n1 * spd + break + k * spd

A tick's match state is the state after the latest delivery whose
estimated time is <= the tick's snapshot time — so attached state is
causal: it never looks at deliveries estimated to be in the future, and
innings-1 length only informs innings-2 ticks (by then it is the past).

Constants were calibrated on this dataset by maximising "wicket-jump
lift": the mean |de-vigged prob move| in the snapshot window around each
estimated wicket time, divided by the baseline mean move. Lift ~2.0x at
the defaults (a perfectly wrong clock would give ~1.0x). Expect a few
balls of error mid-innings (timeouts, drinks, rain); treat attached state
as over-level, not ball-level, truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: minimal per-delivery schema consumed here (ball_by_ball.parquet and
#: btengine.ingest.cricsheet.load_deliveries both provide it)
DELIVERY_COLS = [
    "match_id", "innings", "over", "ball", "innings_team",
    "runs_total", "legal_ball", "wicket",
]

STATE_COLS = [
    "est_innings", "est_balls", "est_legal_balls", "est_over",
    "est_wkts", "est_score", "est_target", "batting_team", "is_batting",
]


@dataclass(frozen=True)
class ClockModel:
    seconds_per_delivery: float = 48.0
    innings_break_min: float = 25.0
    start_offset_min: float = 0.0


DEFAULT_CLOCK = ClockModel()


def delivery_table(bb: pd.DataFrame) -> pd.DataFrame:
    """Order deliveries and add cumulative in-innings state.

    Adds per row (state AFTER the delivery): balls/legal balls bowled,
    score, wickets, and the innings-2 chase target (innings-1 total + 1).
    """
    d = bb[DELIVERY_COLS].sort_values(
        ["match_id", "innings", "over", "ball"], kind="stable"
    ).copy()
    d["runs_total"] = d["runs_total"].astype(float)
    d["legal_ball"] = d["legal_ball"].astype(bool)
    d["wicket"] = d["wicket"].astype(bool)
    grp = d.groupby(["match_id", "innings"], sort=False)
    d["balls"] = grp.cumcount() + 1
    d["legal_balls"] = grp["legal_ball"].cumsum()
    d["score"] = grp["runs_total"].cumsum()
    d["wkts"] = grp["wicket"].cumsum()
    inn1_total = (
        d[d["innings"] == 1].groupby("match_id")["runs_total"].sum()
    )
    d["target"] = np.where(
        d["innings"] == 2, d["match_id"].map(inn1_total) + 1, np.nan
    )
    return d


def estimate_delivery_times(
    deliveries: pd.DataFrame,
    off_time: pd.Timestamp,
    clock: ClockModel = DEFAULT_CLOCK,
) -> pd.Series:
    """Estimated wall-clock time of each delivery of ONE match (state
    becomes known at this time)."""
    d = deliveries
    # k-th delivery bowled overall + one break per innings change (covers
    # super overs: innings 3/4 in tied matches stay monotonic)
    overall = np.arange(1, len(d) + 1)
    secs = (
        clock.start_offset_min * 60
        + overall * clock.seconds_per_delivery
        + (d["innings"].to_numpy() - 1) * clock.innings_break_min * 60
    )
    return pd.Series(
        off_time + pd.to_timedelta(secs, unit="s"), index=d.index
    )


def attach_state(
    ticks: pd.DataFrame,
    bb: pd.DataFrame,
    clock: ClockModel = DEFAULT_CLOCK,
) -> pd.DataFrame:
    """Attach estimated match state (innings, balls, wickets, score,
    target, batting side) to every tick, causally.

    Pre-off ticks and ticks before the estimated first ball get innings 1
    with zero balls/wickets. Matches absent from `bb` get NaN state.
    """
    d = delivery_table(bb)
    out = []
    for match_id, g in ticks.groupby("match_id", sort=False):
        g = g.sort_values("snapshot_time").copy()
        dm = d[d["match_id"] == match_id]
        if dm.empty:
            for c in STATE_COLS:
                g[c] = np.nan
            out.append(g)
            continue
        inplay = g.loc[g["inplay"], "snapshot_time"]
        off = inplay.min()
        dm = dm.assign(est_time=estimate_delivery_times(dm, off, clock))
        state = dm[["est_time", "innings", "balls", "legal_balls",
                    "score", "wkts", "target", "innings_team"]].rename(
            columns={
                "innings": "est_innings", "balls": "est_balls",
                "legal_balls": "est_legal_balls", "score": "est_score",
                "wkts": "est_wkts", "est_target": "est_target",
                "target": "est_target", "innings_team": "batting_team",
            })
        g = pd.merge_asof(
            g, state, left_on="snapshot_time", right_on="est_time",
            direction="backward",
        ).drop(columns="est_time")
        # before the (estimated) first ball: innings 1, nothing happened yet
        first_team = dm.iloc[0]["innings_team"]
        pre = g["est_innings"].isna()
        g.loc[pre, ["est_innings", "est_balls", "est_legal_balls",
                    "est_wkts", "est_score"]] = [1, 0, 0, 0, 0]
        g.loc[pre, "batting_team"] = first_team
        g["est_over"] = g["est_legal_balls"] // 6 + (g["est_legal_balls"] % 6) / 10
        g["is_batting"] = g["team"] == g["batting_team"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def wicket_jump_lift(
    ticks: pd.DataFrame,
    bb: pd.DataFrame,
    clock: ClockModel = DEFAULT_CLOCK,
) -> tuple[float, int]:
    """Alignment diagnostic: mean |prob move| of the batting team in the
    snapshot window around each estimated wicket time, over the baseline
    mean |prob move|. ~1.0 means the clock is uninformative; calibrated
    defaults reach ~2.0 on the kingmaker data."""
    d = delivery_table(bb)
    ip = ticks[ticks["inplay"]]
    jumps, bases = [], []
    for match_id, g in ip.groupby("match_id"):
        dm = d[(d["match_id"] == match_id) & (d["innings"] <= 2)]
        if dm.empty:
            continue
        piv = g.pivot_table(
            index="snapshot_time", columns="team", values="prob"
        ).sort_index().ffill()
        if piv.index.tz is not None:
            piv.index = piv.index.tz_localize(None)
        times = piv.index.values
        base = float(np.nanmean(piv.diff().abs().to_numpy()))
        dm = dm.assign(est_time=estimate_delivery_times(dm, piv.index[0], clock))
        for _, r in dm[dm["wicket"]].iterrows():
            i = int(np.searchsorted(times, np.datetime64(r["est_time"])))
            if not (1 <= i < len(times) - 1) or r["innings_team"] not in piv:
                continue
            move = abs(
                piv[r["innings_team"]].iloc[i + 1]
                - piv[r["innings_team"]].iloc[i - 1]
            )
            if np.isfinite(move):
                jumps.append(move)
                bases.append(base)
    if not jumps:
        return float("nan"), 0
    return float(np.mean(jumps) / np.mean(bases)), len(jumps)
