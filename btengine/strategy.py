"""Declarative strategy specs and the built-in strategy library.

A Strategy is a rule, not code-with-state:
    selection -> pre-off filter -> entry trigger -> side -> time window
    (+ optional early exit for trading strategies)

Entry triggers receive a single tick (pandas Series) with settlement
columns removed by the engine — they cannot see the future or the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import pandas as pd

Selector = Union[str, Callable[[pd.DataFrame], pd.Series]]


@dataclass(frozen=True)
class Strategy:
    """Declarative spec.

    select        which runner per match: "fav" | "dog" | "bats_first" |
                  "bats_second" | callable(ticks) -> boolean mask
    side          "back" or "lay"
    entry         callable(tick) -> bool; tick has no settlement columns
    preoff_filter optional callable(preoff_info dict) -> bool, evaluated on
                  pre-off-knowable fields only: team, preoff_price, is_fav,
                  bats_first
    window        (min, max) minutes-since-off in which entry may trigger
    exit          optional callable(tick, entry_odds) -> bool; close the
                  position (green-up hedge) at the first qualifying tick
                  after entry instead of holding to settlement
    """

    name: str
    select: Selector
    side: str
    entry: Callable[[pd.Series], bool]
    preoff_filter: Optional[Callable[[dict], bool]] = None
    window: Tuple[float, float] = (0.0, math.inf)
    exit: Optional[Callable[[pd.Series, float], bool]] = None

    def __post_init__(self):
        if self.side not in ("back", "lay"):
            raise ValueError(f"side must be 'back' or 'lay', got {self.side!r}")


def selection_mask(ticks: pd.DataFrame, select: Selector) -> pd.Series:
    """Boolean mask picking the strategy's runner, from pre-off info only."""
    if callable(select):
        return select(ticks)
    if select == "fav":
        return ticks["is_fav"]
    if select == "dog":
        return ~ticks["is_fav"]
    if select == "bats_first":
        return ticks["bats_first"]
    if select == "bats_second":
        return ~ticks["bats_first"]
    raise ValueError(f"unknown selector {select!r}")


# ---------------------------------------------------------------------------
# Strategy library
# ---------------------------------------------------------------------------

STRATEGIES: list[Strategy] = []


def register(strategy: Strategy) -> Strategy:
    STRATEGIES.append(strategy)
    return strategy


def lay_drifting_fav(
    drift: float = 1.5,
    max_preoff: float = 2.5,
    window: Tuple[float, float] = (1.0, 120.0),
) -> Strategy:
    """Lay the pre-off favourite once it has drifted `drift`x in-play."""
    return Strategy(
        name=f"lay_fav_drift{drift:g}x",
        select="fav",
        side="lay",
        preoff_filter=lambda p: p["preoff_price"] <= max_preoff,
        entry=lambda t: t["inplay"] and t["drift_mult"] >= drift,
        window=window,
    )


def back_drifting_fav(
    drift: float = 1.5,
    max_preoff: float = 2.5,
    window: Tuple[float, float] = (1.0, 120.0),
) -> Strategy:
    """Control: the contrarian side of lay_drifting_fav (settlement sanity
    check — with zero commission/slippage its P&L mirrors the lay)."""
    return Strategy(
        name=f"back_fav_drift{drift:g}x_CONTROL",
        select="fav",
        side="back",
        preoff_filter=lambda p: p["preoff_price"] <= max_preoff,
        entry=lambda t: t["inplay"] and t["drift_mult"] >= drift,
        window=window,
    )


def lay_steaming_dog(
    steam: float = 0.7,
    window: Tuple[float, float] = (1.0, 120.0),
) -> Strategy:
    """Lay the dog once its price has shortened to `steam`x its pre-off price."""
    return Strategy(
        name=f"lay_dog_steam{steam:g}x",
        select="dog",
        side="lay",
        entry=lambda t: t["inplay"] and t["drift_mult"] <= steam,
        window=window,
    )


def lay_band_drift(
    low: float = 1.70,
    high: float = 1.90,
    trigger: float = 2.30,
    disaster_sl: float = 2.0,
) -> Strategy:
    """PRE-REGISTERED 2026 OOS CANDIDATE (frozen 2026-06-10, before any
    2026 odds were available; tuned on 2021-25 kingmaker data).

    Lay the pre-off favourite priced in [low, high] at the first in-play
    tick >= trigger. No take-profit; disaster stop only: green up if the
    price shortens to entry/disaster_sl (locked loss ~= one stake).
    2021-25 in-sample: +16.4% ROI, max DD ~10 stakes, n=136."""
    return Strategy(
        name=f"lay_fav{low:g}-{high:g}_at{trigger:g}_SL100",
        select="fav",
        side="lay",
        preoff_filter=lambda p: low <= p["preoff_price"] <= high,
        entry=lambda t: bool(t["inplay"] and t["odds"] >= trigger),
        window=(0.0, 600.0),
        exit=lambda t, eo: bool(t["inplay"] and t["odds"] <= eo / disaster_sl),
    )


register(lay_drifting_fav(1.3))
register(lay_drifting_fav(1.5))
register(lay_drifting_fav(2.0))
register(back_drifting_fav(1.5))
register(lay_steaming_dog(0.7))
