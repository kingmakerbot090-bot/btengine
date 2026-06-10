import pytest

from btengine.engine import fill_odds, hedged_pnl, pnl


def test_back_pnl_hand_computed():
    # back 3.0 for 1u, 5% commission: win +2*0.95, lose -1
    assert pnl("back", 3.0, won=True, commission=0.05) == pytest.approx(1.9)
    assert pnl("back", 3.0, won=False, commission=0.05) == pytest.approx(-1.0)


def test_lay_pnl_hand_computed():
    # lay 3.0 for 1u, 5% commission: runner loses +0.95, runner wins -2
    assert pnl("lay", 3.0, won=False, commission=0.05) == pytest.approx(0.95)
    assert pnl("lay", 3.0, won=True, commission=0.05) == pytest.approx(-2.0)


def test_slippage_worsens_both_sides():
    assert fill_odds("back", 2.0, 0.02) == pytest.approx(1.96)
    assert fill_odds("lay", 2.0, 0.02) == pytest.approx(2.04)
    assert pnl("back", 2.0, True, slippage=0.02) == pytest.approx(0.96)
    assert pnl("lay", 2.0, True, slippage=0.02) == pytest.approx(-1.04)


def test_back_and_lay_are_mirrors_without_costs():
    for odds in (1.5, 2.4, 6.0):
        for won in (True, False):
            assert pnl("back", odds, won) == pytest.approx(-pnl("lay", odds, won))


def test_hedged_pnl_is_result_independent():
    # back at 3.0, lay off at 2.0: green-up locks (3/2 - 1) = +0.5
    assert hedged_pnl("back", 3.0, 2.0) == pytest.approx(0.5)
    # lay at 2.0, back off at 3.0: locks (1 - 2/3)
    assert hedged_pnl("lay", 2.0, 3.0) == pytest.approx(1 / 3)
    # losing close: no commission applied
    assert hedged_pnl("back", 2.0, 3.0, commission=0.05) == pytest.approx(-1 / 3)


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        pnl("hedge", 2.0, True)
