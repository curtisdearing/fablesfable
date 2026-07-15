"""Player condition book: per-player performance splits under game conditions.

Built offline by analysis/build_condition_book.py (2019-2025 player-weeks,
empirical-Bayes shrunk toward each player's own base rate, k=25). This module
is the LIVE consumer: it loads book/player_condition_book.{parquet,csv} and
surfaces flagged edges for the context panel.

DISPLAY-ONLY BY CONSTRUCTION (PROP_SHORTLISTER_SPEC.md §4): like the other
panel_items providers, nothing here can touch the composite score or the ML
ranking. Promotion to a scoring factor goes through context_learning tags +
the CLV killcheck, not through this module.

Grading convention behind the numbers: "over" = actual > player's trailing
mean (synthetic line), so edges are directional tendencies, not price edges.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK_PARQUET = os.path.join(ROOT, "book", "player_condition_book.parquet")
BOOK_CSV = os.path.join(ROOT, "book", "player_condition_book.csv")

MIN_N = 6            # career games under the condition before we show it
MIN_EDGE_PP = 4.0    # shrunk percentage-point edge vs the player's own base

_cache: Optional[pd.DataFrame] = None


def load_book(refresh: bool = False) -> pd.DataFrame:
    """Load the book (parquet preferred, CSV fallback); empty frame if absent."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    cols = ["player_id", "condition", "stat", "n", "raw_over", "own_base",
            "shrunk_over", "edge_pp", "flag"]
    if os.path.exists(BOOK_PARQUET):
        df = pd.read_parquet(BOOK_PARQUET)
    elif os.path.exists(BOOK_CSV):
        df = pd.read_csv(BOOK_CSV)
    else:
        df = pd.DataFrame(columns=cols)
    _cache = df
    return df


def edges_for(player_id: str, min_n: int = MIN_N,
              min_edge_pp: float = MIN_EDGE_PP) -> pd.DataFrame:
    """All flagged condition edges for one player, strongest first."""
    book = load_book()
    if book.empty:
        return book
    d = book[(book["player_id"] == player_id) & (book["n"] >= min_n)
             & (book["edge_pp"].abs() >= min_edge_pp)]
    return d.reindex(d["edge_pp"].abs().sort_values(ascending=False).index)


def panel_items(lean: Dict) -> List[str]:
    """Context-panel lines for a lean (same contract as context_features/
    chemistry/ftn panel providers). Empty list when nothing is flagged."""
    pid = lean.get("player_id")
    if not pid:
        return []
    try:
        d = edges_for(str(pid))
    except Exception:                     # book unreadable -> stay silent
        return []
    items: List[str] = []
    for r in d.head(3).itertuples():
        direction = "OVER-leaning" if r.edge_pp > 0 else "UNDER-leaning"
        items.append(
            f"condition book: {r.condition} → {r.stat} {direction} "
            f"({r.raw_over:.0%} vs {r.own_base:.0%} base, n={int(r.n)}, "
            f"shrunk edge {r.edge_pp:+.1f}pp)")
    return items
