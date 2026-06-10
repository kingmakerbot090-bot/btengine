"""btengine — leakage-free backtesting for 2-runner match-odds strategies.

Layers (see README):
    ingest    raw odds/results loading + devig
    ticks     the off-anchored, de-vigged tick table (single source of truth)
    strategy  declarative Strategy spec + selection + library
    engine    causal execution: backtest, pnl
    evaluate  scorecards: evaluate, equity_curve, by_segment, run_strategies, sweep
    report    betlog persistence
"""

from btengine.ingest.odds import devig, load_odds, load_results, load_context
from btengine.ticks import (
    build_tick_table,
    detect_off,
    preoff_reference,
    load_tick_table,
)
from btengine.strategy import Strategy, selection_mask, register, STRATEGIES
from btengine.engine import backtest, pnl, fill_odds
from btengine.evaluate import (
    Result,
    evaluate,
    equity_curve,
    max_drawdown,
    by_segment,
    run_strategies,
    sweep,
)
from btengine.report import save_betlog

__all__ = [
    "devig", "load_odds", "load_results", "load_context",
    "build_tick_table", "detect_off", "preoff_reference", "load_tick_table",
    "Strategy", "selection_mask", "register", "STRATEGIES",
    "backtest", "pnl", "fill_odds",
    "Result", "evaluate", "equity_curve", "max_drawdown",
    "by_segment", "run_strategies", "sweep",
    "save_betlog",
]
