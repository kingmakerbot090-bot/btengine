"""CLI: build the tick table, rank the strategy library, stress execution.

    btengine --odds data/odds.csv --results data/results.csv \
             --context data/context.csv --rebuild
    btengine --demo            # synthetic data, no files needed
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from btengine.evaluate import by_segment, run_strategies
from btengine.engine import backtest
from btengine.report import save_betlog
from btengine.strategy import STRATEGIES
from btengine.synthetic import make_synthetic
from btengine.ticks import build_tick_table, load_tick_table

FMT = {
    "win_rate": "{:.1%}".format, "ci_low": "{:.1%}".format,
    "ci_high": "{:.1%}".format, "breakeven": "{:.1%}".format,
    "roi": "{:+.1%}".format, "total_pnl": "{:+.2f}".format,
    "pnl_per_bet": "{:+.4f}".format, "avg_odds": "{:.2f}".format,
    "max_dd": "{:.2f}".format,
}


def _print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n== {title} ==")
    if table.empty:
        print("(no bets)")
        return
    print(table.to_string(index=False, formatters=FMT, na_rep="-"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="btengine",
        description="Backtest 2-runner match-odds strategies, leakage-free.",
    )
    ap.add_argument("--odds", help="tidy odds CSV (see README data contract)")
    ap.add_argument("--results", help="results CSV: match_id, winner")
    ap.add_argument("--context", help="optional context CSV: match_id, bats_first[, season]")
    ap.add_argument("--cache", default="data/ticks.pkl", help="tick-table cache path")
    ap.add_argument("--rebuild", action="store_true", help="ignore the tick cache")
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--slippage", type=float, default=0.0)
    ap.add_argument("--stress-slippage", type=float, default=0.02,
                    help="extra slippage for the execution stress run")
    ap.add_argument("--save-betlogs", metavar="DIR",
                    help="write one betlog CSV per strategy into DIR")
    ap.add_argument("--kingmaker", metavar="DIR",
                    help="load the kingmaker parquet dump from DIR")
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic data (no input files needed)")
    args = ap.parse_args(argv)

    if args.kingmaker:
        from btengine.align import attach_state, wicket_jump_lift
        from btengine.ingest.kingmaker import (
            load_kingmaker, load_kingmaker_deliveries,
        )
        odds, results, context = load_kingmaker(args.kingmaker)
        ticks = build_tick_table(odds, results, context)
        deliveries = load_kingmaker_deliveries(args.kingmaker)
        lift, n_wkts = wicket_jump_lift(ticks, deliveries)
        print(f"ball-by-ball alignment: wicket-jump lift {lift:.2f}x "
              f"over {n_wkts} wickets (1.0 = uninformative clock)")
        ticks = attach_state(ticks, deliveries)
    elif args.demo:
        odds, results, context = make_synthetic()
        ticks = build_tick_table(odds, results, context)
        print("demo mode: synthetic data (efficient market — expect ~-EV "
              "after commission)")
    elif args.odds and args.results:
        ticks = load_tick_table(
            args.odds, args.results, args.context,
            cache_path=args.cache, rebuild=args.rebuild,
        )
    else:
        ap.error("provide --odds and --results, or use --demo")

    n_matches = ticks["match_id"].nunique()
    print(f"tick table: {len(ticks)} ticks, {n_matches} matches, "
          f"seasons {sorted(ticks['season'].unique())}")

    table = run_strategies(
        STRATEGIES, ticks,
        commission=args.commission, slippage=args.slippage,
    )
    _print_table(
        f"ranked strategies (commission={args.commission:.1%}, "
        f"slippage={args.slippage:.1%})",
        table,
    )

    stress = args.slippage + args.stress_slippage
    stressed = run_strategies(
        STRATEGIES, ticks, commission=args.commission, slippage=stress,
    )
    _print_table(f"execution stress (slippage={stress:.1%})", stressed)

    if not table.empty:
        top_name = table.iloc[0]["name"]
        top = next(s for s in STRATEGIES if s.name == top_name)
        betlog = backtest(top, ticks, args.commission, args.slippage)
        _print_table(f"per-season stability: {top_name}", by_segment(betlog))

    if args.save_betlogs:
        for strategy in STRATEGIES:
            betlog = backtest(strategy, ticks, args.commission, args.slippage)
            path = save_betlog(
                betlog, f"{args.save_betlogs}/{strategy.name}.csv"
            )
            print(f"saved {len(betlog):4d} bets -> {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
