"""Synthetic Betfair-shaped fixtures.

Generates odds/results/context tables matching the data contract so the
whole engine runs end-to-end before real exchange data is plugged in, and
so tests have a controlled world. The in-play probability path is a
martingale and the winner is sampled from the *final* in-play probability,
so the quoted price is the true conditional win probability at every tick:
an efficient market by construction. Every strategy should therefore read
~0 EV before costs and -EV after commission — a useful null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = ["CSK", "MI", "RCB", "KKR", "SRH", "RR", "DC", "PBKS", "GT", "LSG"]


def make_synthetic(
    n_matches: int = 150,
    seasons: tuple[int, ...] = (2023, 2024, 2025),
    snapshots_inplay: int = 60,
    snapshot_mins: float = 3.0,
    overround: float = 1.0,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (odds, results, context) per the ingest data contract.

    overround=1.0 models last-traded prices (fair, like Betfair historic
    data). Raise it to test de-vigging, but note a single quoted price
    with overround systematically flatters one side: backs pay it, lays
    collect it.
    """
    rng = np.random.default_rng(seed)
    odds_rows, results_rows, context_rows = [], [], []

    for i in range(n_matches):
        match_id = f"M{i:04d}"
        season = seasons[i % len(seasons)]
        home, away = rng.choice(TEAMS, size=2, replace=False)
        off = pd.Timestamp(f"{season}-04-01 14:00:00") + pd.Timedelta(days=int(i))

        p0 = rng.uniform(0.40, 0.78)            # home's true pre-off prob

        # Pre-off quotes: 3 snapshots, small noise around p0.
        for k in range(3):
            t = off - pd.Timedelta(minutes=30 - 10 * k)
            p = float(np.clip(p0 + rng.normal(0, 0.01), 0.05, 0.95))
            for team, prob in ((home, p), (away, 1 - p)):
                odds_rows.append(
                    (match_id, t, team, overround_price(prob, overround), False)
                )

        # In-play: martingale random walk in probability space. The winner
        # is sampled from the final probability, so E[win | p_t] = p_t and
        # no tick offers an edge.
        p = p0
        for k in range(snapshots_inplay):
            p = float(np.clip(p + rng.normal(0, 0.045), 0.02, 0.98))
            t = off + pd.Timedelta(minutes=k * snapshot_mins)
            for team, prob in ((home, p), (away, 1 - p)):
                odds_rows.append(
                    (match_id, t, team, overround_price(prob, overround), True)
                )
        winner = home if rng.random() < p else away

        results_rows.append((match_id, winner))
        bats_first = home if rng.random() < 0.5 else away
        context_rows.append((match_id, bats_first, season))

    odds = pd.DataFrame(
        odds_rows,
        columns=["match_id", "snapshot_time", "team", "decimal_odds", "inplay"],
    )
    results = pd.DataFrame(results_rows, columns=["match_id", "winner"])
    context = pd.DataFrame(
        context_rows, columns=["match_id", "bats_first", "season"]
    )
    return odds, results, context


def overround_price(prob: float, overround: float) -> float:
    """Decimal odds for `prob` with the book summing to `overround`."""
    return round(max(1.01, 1.0 / (prob * overround)), 2)
