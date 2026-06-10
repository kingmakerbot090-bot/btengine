import pandas as pd
import pytest

from btengine.engine import backtest
from btengine.strategy import Strategy, lay_drifting_fav, selection_mask
from tests.conftest import T0


def test_entry_at_first_qualifying_tick(tiny_ticks):
    # Fav A hits 1.5x drift (1.50 -> 2.25) at minute 20; B won, so the lay wins.
    betlog = backtest(lay_drifting_fav(1.5), tiny_ticks, commission=0.0)
    assert len(betlog) == 1
    bet = betlog.iloc[0]
    assert bet["team"] == "A"
    assert bet["entry_time"] == T0 + pd.Timedelta(minutes=20)
    assert bet["entry_odds"] == pytest.approx(2.25)
    assert not bet["won"]
    assert bet["pnl"] == pytest.approx(1.0)


def test_back_side_settles_against_us(tiny_ticks):
    strat = Strategy(
        name="back_fav_drift",
        select="fav",
        side="back",
        entry=lambda t: t["drift_mult"] >= 1.5,
        window=(0, 120),
    )
    betlog = backtest(strat, tiny_ticks, commission=0.0)
    assert betlog.iloc[0]["pnl"] == pytest.approx(-1.0)


def test_window_excludes_late_trigger(tiny_ticks):
    # drift first reaches 1.5x at minute 20 — a (0, 15) window must skip it
    betlog = backtest(lay_drifting_fav(1.5, window=(0, 15)), tiny_ticks)
    assert betlog.empty


def test_preoff_filter_blocks_match(tiny_ticks):
    # fav pre-off price is 1.50, filter demands <= 1.2
    betlog = backtest(lay_drifting_fav(1.5, max_preoff=1.2), tiny_ticks)
    assert betlog.empty


def test_selection_masks(tiny_ticks):
    assert selection_mask(tiny_ticks, "fav").equals(tiny_ticks["is_fav"])
    assert selection_mask(tiny_ticks, "dog").equals(~tiny_ticks["is_fav"])
    assert selection_mask(tiny_ticks, "bats_first").equals(
        tiny_ticks["bats_first"]
    )
    with pytest.raises(ValueError):
        selection_mask(tiny_ticks, "nonsense")


def test_exit_closes_early_with_hedged_pnl(tiny_ticks):
    # Lay fav at 1.5x drift (2.25 @ min 20), exit when odds reach 3.0 (min 30):
    # lay-then-back green-up locks 1 - 2.25/3.0 = +0.25 regardless of result.
    strat = Strategy(
        name="lay_fav_with_exit",
        select="fav",
        side="lay",
        entry=lambda t: t["drift_mult"] >= 1.5,
        exit=lambda t, entry_odds: t["odds"] >= entry_odds * 4 / 3,
        window=(0, 120),
    )
    betlog = backtest(strat, tiny_ticks, commission=0.0)
    bet = betlog.iloc[0]
    assert bet["exited"]
    assert bet["exit_odds"] == pytest.approx(3.0)
    assert bet["pnl"] == pytest.approx(0.25)


def test_one_bet_per_match(synth_ticks):
    betlog = backtest(lay_drifting_fav(1.3), synth_ticks)
    assert betlog["match_id"].is_unique
