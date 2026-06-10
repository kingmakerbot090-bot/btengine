"""Causal execution engine: generate bets using only data <= entry instant.

Causality is enforced structurally:
  * candidate ticks are scanned in time order; the first qualifying tick is
    the entry and scanning stops;
  * settlement columns (ticks.RESULT_COLS) are dropped from the view the
    entry/exit triggers see, so a trigger touching the result raises KeyError;
  * pre-off filters see only the pre-off-knowable runner info.
tests/test_leakage.py corrupts every post-entry tick and asserts entries
are unchanged.
"""

from __future__ import annotations

import pandas as pd

from btengine.strategy import Strategy, selection_mask
from btengine.ticks import RESULT_COLS

BETLOG_COLS = [
    "match_id", "team", "side", "entry_time", "entry_mins", "entry_odds",
    "preoff_price", "drift_at_entry", "exited", "exit_odds", "stake",
    "won", "pnl", "season",
]


def fill_odds(side: str, odds: float, slippage: float) -> float:
    """Worsen the quoted price by `slippage` (fraction): backs get shorter
    odds, lays get longer odds."""
    return odds * (1.0 - slippage) if side == "back" else odds * (1.0 + slippage)


def pnl(
    side: str,
    odds: float,
    won: bool,
    commission: float = 0.0,
    slippage: float = 0.0,
    stake: float = 1.0,
) -> float:
    """Flat-stake back/lay payoff, settled on the result.

    `won` is whether the runner won the match. Commission applies to net
    winnings only; slippage worsens the entry price via fill_odds.
    """
    filled = fill_odds(side, odds, slippage)
    if side == "back":
        return (filled - 1.0) * stake * (1.0 - commission) if won else -stake
    if side == "lay":
        return -(filled - 1.0) * stake if won else stake * (1.0 - commission)
    raise ValueError(f"side must be 'back' or 'lay', got {side!r}")


def hedged_pnl(
    side: str,
    entry_odds: float,
    exit_odds: float,
    commission: float = 0.0,
    stake: float = 1.0,
) -> float:
    """Green-up close: equalised P&L from entering at entry_odds and taking
    the opposite side at exit_odds, sized so the result no longer matters."""
    if side == "back":
        gross = stake * (entry_odds / exit_odds - 1.0)
    elif side == "lay":
        gross = stake * (1.0 - entry_odds / exit_odds)
    else:
        raise ValueError(f"side must be 'back' or 'lay', got {side!r}")
    return gross * (1.0 - commission) if gross > 0 else gross


def backtest(
    strategy: Strategy,
    ticks: pd.DataFrame,
    commission: float = 0.02,
    slippage: float = 0.0,
    stake: float = 1.0,
) -> pd.DataFrame:
    """Run one strategy over the tick table -> betlog (one row per bet).

    Enters at the first qualifying tick per match inside the strategy's
    time window; settles on the result, or closes early via the strategy's
    exit rule if it has one.
    """
    lo, hi = strategy.window
    candidates = ticks[selection_mask(ticks, strategy.select)]
    drop = [c for c in RESULT_COLS if c in candidates.columns]

    bets = []
    for match_id, g in candidates.groupby("match_id", sort=False):
        g = g.sort_values("snapshot_time")
        first = g.iloc[0]
        if strategy.preoff_filter is not None:
            preoff_info = {
                "team": first["team"],
                "preoff_price": first["preoff_price"],
                "is_fav": bool(first["is_fav"]),
                "bats_first": bool(first["bats_first"]),
            }
            if not strategy.preoff_filter(preoff_info):
                continue

        in_window = g[(g["mins_since_off"] >= lo) & (g["mins_since_off"] <= hi)]
        visible = in_window.drop(columns=drop)

        entry = None
        for _, tick in visible.iterrows():
            if strategy.entry(tick):
                entry = tick
                break
        if entry is None:
            continue

        won = bool(g.loc[entry.name, "won"])
        entry_odds = fill_odds(strategy.side, entry["odds"], slippage)

        exited, exit_odds, bet_pnl = False, None, None
        if strategy.exit is not None:
            after = visible[visible["snapshot_time"] > entry["snapshot_time"]]
            for _, tick in after.iterrows():
                if strategy.exit(tick, entry_odds):
                    close_side = "lay" if strategy.side == "back" else "back"
                    exit_odds = fill_odds(close_side, tick["odds"], slippage)
                    bet_pnl = hedged_pnl(
                        strategy.side, entry_odds, exit_odds, commission, stake
                    )
                    exited = True
                    break
        if not exited:
            bet_pnl = pnl(
                strategy.side, entry["odds"], won,
                commission=commission, slippage=slippage, stake=stake,
            )

        bets.append({
            "match_id": match_id,
            "team": entry["team"],
            "side": strategy.side,
            "entry_time": entry["snapshot_time"],
            "entry_mins": entry["mins_since_off"],
            "entry_odds": entry_odds,
            "preoff_price": entry["preoff_price"],
            "drift_at_entry": entry["drift_mult"],
            "exited": exited,
            "exit_odds": exit_odds,
            "stake": stake,
            "won": won,
            "pnl": bet_pnl,
            "season": entry["season"],
        })

    return pd.DataFrame(bets, columns=BETLOG_COLS)
