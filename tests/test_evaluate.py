import numpy as np
import pandas as pd
import pytest

from btengine.engine import backtest
from btengine.evaluate import (
    by_segment,
    equity_curve,
    evaluate,
    max_drawdown,
    run_strategies,
    sweep,
    wilson_ci,
)
from btengine.strategy import back_drifting_fav, lay_drifting_fav


def _betlog(pnls, seasons=None):
    n = len(pnls)
    return pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)],
        "entry_time": pd.date_range("2025-04-01", periods=n, freq="D"),
        "entry_odds": [2.0] * n,
        "stake": [1.0] * n,
        "won": [p > 0 for p in pnls],
        "pnl": pnls,
        "season": seasons or [2025] * n,
    })


def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = wilson_ci(60, 100)
    assert lo < 0.6 < hi
    assert (lo, hi) == pytest.approx((0.502, 0.691), abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_evaluate_scorecard():
    res = evaluate("s", _betlog([1.0, 1.0, -1.0, -1.0]))
    assert res.n == 4 and res.wins == 2
    assert res.win_rate == pytest.approx(0.5)
    assert res.total_pnl == pytest.approx(0.0)
    assert res.roi == pytest.approx(0.0)
    assert res.breakeven == pytest.approx(0.5)  # symmetric payoffs
    assert res.ci_low < 0.5 < res.ci_high


def test_evaluate_empty_betlog():
    res = evaluate("empty", pd.DataFrame(columns=_betlog([1.0]).columns))
    assert res.n == 0 and res.total_pnl == 0.0


def test_equity_curve_and_max_drawdown():
    curve = equity_curve(_betlog([1.0, 1.0, -2.0, -1.0, 3.0]))
    assert list(curve) == [1.0, 2.0, 0.0, -1.0, 2.0]
    assert max_drawdown(curve) == pytest.approx(3.0)  # peak 2 -> trough -1
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


def test_max_drawdown_counts_initial_losses():
    curve = equity_curve(_betlog([-1.0, -1.0]))
    assert max_drawdown(curve) == pytest.approx(2.0)


def test_by_segment_splits_per_season():
    log = _betlog([1.0, -1.0, 1.0, 1.0], seasons=[2024, 2024, 2025, 2025])
    seg = by_segment(log, "season")
    assert len(seg) == 2
    assert seg.set_index("season").loc["2025", "total_pnl"] == pytest.approx(2.0)


def test_control_strategy_mirrors_its_inverse(synth_ticks):
    """Settlement sanity check: back vs lay of the same entries, zero costs,
    must be exact mirror images."""
    lay = backtest(lay_drifting_fav(1.5), synth_ticks, commission=0.0)
    back = backtest(back_drifting_fav(1.5), synth_ticks, commission=0.0)
    assert list(lay["match_id"]) == list(back["match_id"])
    assert np.allclose(lay["entry_odds"], back["entry_odds"])
    assert np.allclose(lay["pnl"].to_numpy(), -back["pnl"].to_numpy())
    r_lay = evaluate("lay", lay)
    r_back = evaluate("back", back)
    assert r_lay.total_pnl == pytest.approx(-r_back.total_pnl)
    assert r_lay.wins + r_back.wins == r_lay.n


def test_run_strategies_ranks_by_pnl_per_bet(synth_ticks):
    table = run_strategies(
        [lay_drifting_fav(1.5), back_drifting_fav(1.5)], synth_ticks,
        commission=0.0,
    )
    ranked = table["pnl_per_bet"].to_numpy()
    assert (ranked[:-1] >= ranked[1:]).all()
    assert set(table.columns) >= {"name", "n", "win_rate", "ci_low",
                                  "ci_high", "roi", "max_dd"}


def test_sweep_covers_the_grid(synth_ticks):
    table = sweep(
        lay_drifting_fav,
        {"drift": [1.3, 1.6], "max_preoff": [2.0, 3.0]},
        synth_ticks,
    )
    assert len(table) == 4
    assert {"drift", "max_preoff", "total_pnl"} <= set(table.columns)
