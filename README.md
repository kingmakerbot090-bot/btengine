# btengine — leakage-free betting-strategy backtester (MVP)

Turns timestamped market prices + settled results into honest strategy
performance: win% with confidence intervals, P&L/ROI, drawdown, per-season
stability — with **zero look-ahead leakage**, so a positive result is
trustworthy rather than an artifact.

Scope: one market type (2-runner MATCH_ODDS), one league (IPL). There are
no per-ball timestamps: strategies run on the timestamped odds stream and
settle on the final result. The only in-match clock is **minutes since the
"off"** (first in-play quote) — a live-knowable anchor, never
fraction-of-total-match.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                      # includes the leakage gate
btengine --demo             # end-to-end on synthetic (efficient-market) data
```

With real Betfair data:

```bash
btengine --odds data/odds.csv --results data/results.csv \
         --context data/context.csv --rebuild \
         --commission 0.02 --slippage 0.0 --save-betlogs out/
```

With the kingmaker parquet dump (the zip lives on the `data` branch;
unzip it into `data/`):

```bash
btengine --kingmaker data/kingmaker-data --save-betlogs out/
KINGMAKER_DATA=data/kingmaker-data pytest   # enables integration tests
```

Match facts come from the raw cricsheet dump
(https://cricsheet.org/downloads/ipl_json.zip — the `ipl_json/` folder
in the dump), parsed by `btengine/ingest/cricsheet.py`; the pre-parsed
parquets are only a fallback (`load_kingmaker(..., source=...)`). Tied
matches settle on the super-over winner, matching Betfair MATCH_ODDS
rules; abandoned matches are dropped.

The adapter (`btengine/ingest/kingmaker.py`) maps the quote stream onto
the contract below: keeps only OPEN match-odds runners (the raw stream
mixes in innings-runs and player props), normalises team renames
(Bangalore→Bengaluru), and dedupes re-sent quotes. Yield: 274 settled
IPL matches, 2021–2025, ~227k ticks. Pipeline validation: 528/528
pre-off closing prices match the independently computed
`closing_lines.parquet` exactly, and the cricsheet parser is
cross-checked against the pre-parsed tables (`tests/test_kingmaker.py`).

## Data contract (inputs the engine assumes)

`--odds` — tidy odds table, one row per quote:

| column          | type     | meaning                              |
|-----------------|----------|--------------------------------------|
| `match_id`      | str      | market identifier                    |
| `snapshot_time` | datetime | when the quote was observed          |
| `team`          | str      | runner name                          |
| `decimal_odds`  | float    | decimal price (> 1.0)                |
| `inplay`        | bool     | market in-play at this snapshot      |

`--results` — `match_id, winner`.
`--context` (optional) — `match_id, bats_first[, season]`. Without it,
`bats_first` selectors are unavailable and season falls back to quote year.

Everything downstream is derived from these. Matches missing a result,
in-play quotes, or a pre-off price for both runners are dropped.

## Core abstractions

| object     | is                        | holds |
|------------|---------------------------|-------|
| tick table | the point-in-time price path | per (match, runner, time): odds, de-vigged prob, mins-since-off, pre-off reference price, drift multiple, fav/dog/bats-first tags, result label |
| `Strategy` | a declarative rule        | selection → pre-off filter → entry trigger → side → time window (+ optional exit) |
| betlog     | what the strategy did     | one row per bet: match, side, entry odds/time, won, P&L |
| `Result`   | scorecard of a strategy   | n, win%, Wilson CI, P&L/bet, ROI, drawdown, breakeven |

## Module map

```
btengine/
  ingest/odds.py   devig, load_odds/results/context (the data contract)
  ticks.py         build_tick_table, detect_off, preoff_reference,
                   load_tick_table(rebuild=) — the single source of truth
  strategy.py      Strategy dataclass, selection_mask, register/STRATEGIES,
                   library: lay_drifting_fav, back_drifting_fav (control),
                   lay_steaming_dog
  engine.py        backtest (causal execution), pnl, fill_odds, hedged exits
  evaluate.py      evaluate (Wilson CI, breakeven), equity_curve,
                   max_drawdown, by_segment, run_strategies, sweep
  report.py        save_betlog
  synthetic.py     Betfair-shaped martingale fixtures (efficient-market null)
  cli.py           btengine --demo | --odds/--results [--rebuild ...]
```

## Defining a strategy

```python
from btengine import Strategy, backtest, evaluate

strat = Strategy(
    name="lay_fav_drift1.5x",
    select="fav",                       # from pre-off info only
    side="lay",
    preoff_filter=lambda p: p["preoff_price"] <= 2.5,
    entry=lambda t: t["inplay"] and t["drift_mult"] >= 1.5,
    window=(1, 120),                    # minutes since the off
    # exit=lambda t, entry_odds: ...    # optional green-up close
)
betlog = backtest(strat, ticks, commission=0.02)
print(evaluate(strat.name, betlog))
```

Entry triggers receive one tick at a time, in time order, with settlement
columns **removed** — a trigger that touches `won` raises `KeyError`.

## Integrity guards (what makes the numbers trustworthy)

1. **Causality, structurally**: entries scan ticks in time order and stop at
   the first qualifying tick; triggers cannot see the result column.
2. **Causality, tested**: `tests/test_leakage.py` corrupts every post-entry
   tick and asserts the betlog is unchanged. This test gates every number.
3. **Live-knowable clock**: minutes-since-off only.
4. **Control strategy**: `back_drifting_fav` is the contrarian mirror of
   `lay_drifting_fav`; with zero costs their P&Ls are exact negatives
   (tested) — a settlement sanity check.
5. **Efficient-market null**: the synthetic generator prices a martingale
   and samples the winner from the final price, so every strategy should
   read ~0 EV before costs there. An engine that finds an edge in the demo
   data is broken.

## Costs model

Flat stakes (MVP). `commission` is charged on net winnings only.
`slippage` worsens the entry price: backs fill at `odds*(1-s)`, lays at
`odds*(1+s)`. The CLI prints a second table at stressed slippage
(`--stress-slippage`, default +2%) so fragile edges are visible.
