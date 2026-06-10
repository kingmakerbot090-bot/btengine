import numpy as np
import pandas as pd
import pytest

from btengine.align import (
    ClockModel,
    attach_state,
    delivery_table,
    estimate_delivery_times,
)

T0 = pd.Timestamp("2025-04-01 14:00:00")
CLOCK = ClockModel(seconds_per_delivery=60.0, innings_break_min=2.0,
                   start_offset_min=0.0)


@pytest.fixture()
def tiny_bb():
    """m1: A bats 12 deliveries (2 overs, wicket on 1.2, 10 runs),
    then B bats 6 deliveries."""
    rows = []
    for over in (0, 1):
        for ball in range(1, 7):
            rows.append(("m1", 1, over, ball, "A", 1 if over == 0 else 0,
                         True, over == 1 and ball == 2))
    for ball in range(1, 7):
        rows.append(("m1", 2, 0, ball, "B", 2, True, False))
    return pd.DataFrame(rows, columns=[
        "match_id", "innings", "over", "ball", "innings_team",
        "runs_total", "legal_ball", "wicket",
    ])


@pytest.fixture()
def tiny_align_ticks():
    times = {
        "preoff": T0 - pd.Timedelta(minutes=5),
        "off": T0,
        "mid1": T0 + pd.Timedelta(minutes=5, seconds=30),
        "break": T0 + pd.Timedelta(minutes=13),
        "chase": T0 + pd.Timedelta(minutes=15, seconds=30),
    }
    rows = []
    for label, t in times.items():
        for team in ("A", "B"):
            rows.append(("m1", t, team, label != "preoff", label))
    return pd.DataFrame(
        rows, columns=["match_id", "snapshot_time", "team", "inplay", "label"]
    )


def test_delivery_table_cumulative_state(tiny_bb):
    d = delivery_table(tiny_bb)
    last1 = d[d["innings"] == 1].iloc[-1]
    assert last1["balls"] == 12 and last1["score"] == 6 and last1["wkts"] == 1
    assert (d.loc[d["innings"] == 2, "target"] == 7).all()
    assert d[d["innings"] == 1]["target"].isna().all()


def test_estimated_times_follow_the_clock(tiny_bb):
    d = delivery_table(tiny_bb)
    dm = d[d["match_id"] == "m1"]
    est = estimate_delivery_times(dm, T0, CLOCK)
    # innings 1 delivery k at off + k minutes
    assert est.iloc[0] == T0 + pd.Timedelta(minutes=1)
    assert est.iloc[11] == T0 + pd.Timedelta(minutes=12)
    # innings 2 starts after 12 deliveries + 2 min break
    assert est.iloc[12] == T0 + pd.Timedelta(minutes=15)


def test_attach_state_is_causal_and_correct(tiny_bb, tiny_align_ticks):
    out = attach_state(tiny_align_ticks, tiny_bb, CLOCK).set_index(
        ["label", "team"]
    )
    pre = out.loc[("preoff", "A")]
    assert pre["est_innings"] == 1 and pre["est_balls"] == 0
    mid = out.loc[("mid1", "A")]  # 5.5 min in -> 5 deliveries done
    assert mid["est_balls"] == 5 and mid["est_wkts"] == 0
    assert mid["is_batting"] and not out.loc[("mid1", "B")]["is_batting"]
    brk = out.loc[("break", "A")]  # innings 1 complete, innings 2 not begun
    assert brk["est_innings"] == 1 and brk["est_balls"] == 12
    assert brk["est_wkts"] == 1 and brk["est_score"] == 6
    chase = out.loc[("chase", "B")]  # 15.5 min -> 1 ball into the chase
    assert chase["est_innings"] == 2 and chase["est_balls"] == 1
    assert chase["est_target"] == 7 and chase["is_batting"]


def test_future_deliveries_cannot_change_past_state(tiny_bb, tiny_align_ticks):
    full = attach_state(tiny_align_ticks, tiny_bb, CLOCK)
    truncated = attach_state(
        tiny_align_ticks, tiny_bb[tiny_bb["innings"] == 1], CLOCK
    )
    early = tiny_align_ticks["label"] != "chase"
    cols = ["est_innings", "est_balls", "est_wkts", "est_score"]
    pd.testing.assert_frame_equal(
        full.loc[early.values, cols], truncated.loc[early.values, cols]
    )


def test_match_without_deliveries_gets_nan_state(tiny_align_ticks):
    empty = pd.DataFrame(columns=[
        "match_id", "innings", "over", "ball", "innings_team",
        "runs_total", "legal_ball", "wicket",
    ])
    out = attach_state(tiny_align_ticks, empty, CLOCK)
    assert out["est_innings"].isna().all()
