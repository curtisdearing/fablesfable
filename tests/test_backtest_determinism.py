"""Backtest determinism: seeded sims reproduce, per-game seeds are stable and
order-independent (regression for the unseeded-MC find, 2026-07-30)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import derive_seed, DEFAULT_SEED
from nflvalue import montecarlo as mc

import json

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "league_priors.json")) as _fh:
    PRIORS = json.load(_fh)          # tracked in-repo; same file backtest.py uses
HOME = {"off": 2.0, "def": 1.0}
AWAY = {"off": -1.0, "def": 0.5}


def test_simulate_reproduces_with_same_seed():
    a = mc.simulate(HOME, AWAY, PRIORS, spread_line=3.5, total_line=44.5,
                    n=500, seed=123)
    b = mc.simulate(HOME, AWAY, PRIORS, spread_line=3.5, total_line=44.5,
                    n=500, seed=123)
    assert a == b


def test_simulate_differs_across_seeds():
    a = mc.simulate(HOME, AWAY, PRIORS, spread_line=3.5, total_line=44.5,
                    n=500, seed=123)
    b = mc.simulate(HOME, AWAY, PRIORS, spread_line=3.5, total_line=44.5,
                    n=500, seed=124)
    assert a != b


def test_derive_seed_is_stable_and_identity_keyed():
    g1 = {"season": 2024, "week": 7, "home": "KC", "away": "BUF"}
    g2 = {"season": 2024, "week": 7, "home": "BUF", "away": "KC"}
    assert derive_seed(g1) == derive_seed(dict(g1))          # stable
    assert derive_seed(g1) != derive_seed(g2)                # identity-keyed
    assert derive_seed(g1, base=DEFAULT_SEED + 1) != derive_seed(g1)
    assert 0 <= derive_seed(g1) < 2 ** 32


def test_derive_seed_order_independent():
    """A game's seed must not depend on where it sits in the file."""
    games = [{"season": 2023, "week": w, "home": "DAL", "away": "PHI"}
             for w in range(1, 6)]
    forward = [derive_seed(g) for g in games]
    backward = [derive_seed(g) for g in reversed(games)]
    assert forward == list(reversed(backward))
