"""Evaluation & comparison: scorecards, equity/drawdown, segments, sweeps."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from btengine.engine import backtest
from btengine.strategy import Strategy


@dataclass(frozen=True)
class Result:
    """Scorecard of a strategy over a betlog. win = bet profitable
    (pnl > 0), so lay wins count as wins. breakeven is the empirical
    win rate needed for zero EV at the observed payoff sizes."""

    name: str
    n: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    total_pnl: float
    pnl_per_bet: float
    roi: float
    avg_odds: float
    breakeven: float
    max_dd: float

    def to_dict(self) -> dict:
        return asdict(self)


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def equity_curve(betlog: pd.DataFrame) -> pd.Series:
    """Cumulative P&L in entry-time order, indexed by entry_time."""
    if betlog.empty:
        return pd.Series(dtype=float)
    ordered = betlog.sort_values("entry_time")
    return pd.Series(
        ordered["pnl"].cumsum().to_numpy(),
        index=ordered["entry_time"],
        name="equity",
    )


def max_drawdown(curve: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve (>= 0)."""
    if curve.empty:
        return 0.0
    running_peak = np.maximum.accumulate(np.maximum(curve.to_numpy(), 0.0))
    return float((running_peak - curve.to_numpy()).max())


def evaluate(name: str, betlog: pd.DataFrame) -> Result:
    """Scorecard: n, win% with Wilson CI, P&L total/per-bet (ROI on flat
    stakes), avg entry odds, empirical breakeven win rate, max drawdown."""
    n = len(betlog)
    if n == 0:
        return Result(name, 0, 0, float("nan"), 0.0, 1.0,
                      0.0, float("nan"), float("nan"), float("nan"),
                      float("nan"), 0.0)
    won_mask = betlog["pnl"] > 0
    wins = int(won_mask.sum())
    ci_low, ci_high = wilson_ci(wins, n)
    total = float(betlog["pnl"].sum())
    staked = float(betlog["stake"].sum())
    avg_win = float(betlog.loc[won_mask, "pnl"].mean()) if wins else 0.0
    avg_loss = float(-betlog.loc[~won_mask, "pnl"].mean()) if wins < n else 0.0
    breakeven = (
        avg_loss / (avg_loss + avg_win)
        if (avg_loss + avg_win) > 0 else float("nan")
    )
    return Result(
        name=name,
        n=n,
        wins=wins,
        win_rate=wins / n,
        ci_low=ci_low,
        ci_high=ci_high,
        total_pnl=total,
        pnl_per_bet=total / n,
        roi=total / staked if staked else float("nan"),
        avg_odds=float(betlog["entry_odds"].mean()),
        breakeven=breakeven,
        max_dd=max_drawdown(equity_curve(betlog)),
    )


def by_segment(betlog: pd.DataFrame, key: str = "season") -> pd.DataFrame:
    """Scorecard per segment (e.g. per season) — stability over time."""
    rows = [
        evaluate(str(seg), seg_log).to_dict()
        for seg, seg_log in betlog.groupby(key)
    ]
    out = pd.DataFrame(rows)
    return out.rename(columns={"name": key}) if not out.empty else out


def run_strategies(
    strategies: Sequence[Strategy],
    ticks: pd.DataFrame,
    commission: float = 0.02,
    slippage: float = 0.0,
    stake: float = 1.0,
) -> pd.DataFrame:
    """Backtest and rank many strategies; ranked by P&L per bet (ROI)."""
    rows = []
    for strategy in strategies:
        betlog = backtest(strategy, ticks, commission, slippage, stake)
        rows.append(evaluate(strategy.name, betlog).to_dict())
    table = pd.DataFrame(rows)
    return table.sort_values(
        "pnl_per_bet", ascending=False, na_position="last"
    ).reset_index(drop=True)


def sweep(
    factory: Callable[..., Strategy],
    param_grid: Mapping[str, Iterable],
    ticks: pd.DataFrame,
    commission: float = 0.02,
    slippage: float = 0.0,
    stake: float = 1.0,
) -> pd.DataFrame:
    """Robustness across thresholds: build a strategy per grid point via
    `factory(**params)`, backtest each, return params + scorecard rows."""
    keys = list(param_grid)
    rows = []
    for values in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        strategy = factory(**params)
        betlog = backtest(strategy, ticks, commission, slippage, stake)
        rows.append({**params, **evaluate(strategy.name, betlog).to_dict()})
    return pd.DataFrame(rows)
