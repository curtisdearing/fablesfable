"""Collect every measured accept-gate verdict into one honest registry.

The dashboard's Honest Record tab shows not just the record, but WHAT WAS
MEASURED AND REJECTED — negative results are load-bearing in this repo, and
hiding them would misrepresent how much of the model survived contact with
its own gates.  Entries come from the machine-written books under ``book/``
wherever one exists; verdicts that predate the books are carried as static
entries pointing at their decision-log write-ups.

Each entry: name, scope, verdict (shipped | rejected | retained | research_only),
numbers (one-line measured summary), source, date.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from . import config

BOOK_DIR = os.path.join(config.ROOT, "book")


def _load(name):
    try:
        with open(os.path.join(BOOK_DIR, name)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _cover_calibration() -> List[Dict]:
    book = _load("cover_calibration.json")
    if not book:
        return []
    out = []
    layers = book.get("rating_predictor_layers") or {}
    if "gaussian_fit" in layers and "empirical_integer" in layers:
        out.append({
            "name": "Key-number kernel (M-G1)",
            "scope": "game spread cover probability",
            "verdict": "rejected",
            "numbers": (f"cover Brier {layers['empirical_integer']} vs gaussian "
                        f"{layers['gaussian_fit']}; needed -0.002, measured worse"),
            "source": "book/cover_calibration.json",
            "date": "2026-07-15",
        })
    sim = book.get("sim_tail_vs_gaussian") or {}
    if sim.get("sim_tail") is not None and sim.get("gaussian_on_sim_mean") is not None:
        kept = sim["sim_tail"] <= sim["gaussian_on_sim_mean"]
        out.append({
            "name": "Sim tail vs fitted gaussian (M-G2)",
            "scope": "game spread cover probability",
            "verdict": "retained" if kept else "rejected",
            "numbers": (f"sim tail Brier {sim['sim_tail']} vs gaussian "
                        f"{sim['gaussian_on_sim_mean']} on real dumped predictions; "
                        + ("sim tail stays in EV math" if kept else "gaussian replaces sim tail")),
            "source": "book/cover_calibration.json",
            "date": "2026-07-30",
        })
    return out


def _fair_value() -> List[Dict]:
    book = _load("fair_value.json")
    if not book:
        return []
    out = []
    for market, res in (book.get("markets") or {}).items():
        pooled, gate = res.get("pooled") or {}, res.get("gate") or {}
        shipped = bool(gate.get("passed"))
        out.append({
            "name": f"Fair-value market blend ({market})",
            "scope": "price context (never a side-picker)",
            "verdict": "shipped" if shipped else "rejected",
            "numbers": (f"walk-forward MAE blend {pooled.get('mae_blend')} vs market "
                        f"{pooled.get('mae_market')}; P(beat market) "
                        f"{pooled.get('p_blend_beats_market')} vs 0.90 gate"),
            "source": "book/fair_value.json",
            "date": "2026-07-30",
        })
    return out


def _loc_features() -> List[Dict]:
    book = _load("loc_features_eval.json")
    if not book:
        return []
    seeds = book.get("seeds") or {}
    gate = book.get("gate") or {}
    deltas = ", ".join(f"seed {k}: ll {v.get('ll_delta_pooled'):+} "
                       f"(P {v.get('p_improve_ll')})" for k, v in seeds.items())
    return [{
        "name": "pass_location features (loc shares + matchup EPA)",
        "scope": "prop ranker features",
        "verdict": "shipped" if gate.get("passed") else "rejected",
        "numbers": f"walk-forward 2021-2024 A/B vs lean set: {deltas}; gate 0.90",
        "source": "book/loc_features_eval.json",
        "date": "2026-07-30",
    }]


def _qb_haircut() -> List[Dict]:
    book = _load("qb_haircut.json")
    if not book:
        return []
    out = []
    for key, v in (book.get("variants") or {}).items():
        pooled, gate = v.get("pooled") or {}, v.get("gate") or {}
        diag = v.get("flagged_diagnostics") or {}
        shipped = bool(gate.get("passed"))
        out.append({
            "name": f"QB backup-start haircut ({v.get('detection_mode', key)})",
            "scope": "game margin forecast",
            "verdict": "shipped" if shipped else "research_only",
            "numbers": (f"backup teams underperform sim by "
                        f"{diag.get('mean_signed_residual_vs_backup')} pts "
                        f"(n={diag.get('n_one_sided_backup')}), but pooled "
                        f"P(improve) {pooled.get('p_adj_beats_base')} vs 0.90 gate"),
            "source": "book/qb_haircut.json",
            "date": "2026-07-30",
        })
    return out


def _bayes_projection() -> List[Dict]:
    book = _load("bayes_projection_eval.json")
    if not book:
        return []
    gate = book.get("gate") or {}
    seeds = book.get("seeds") or {}
    bits = []
    for seed, v in seeds.items():
        p = (v.get("primary") or {}).get("pooled") or {}
        bits.append(
            f"seed {seed}: CRPS {p.get('crps_incumbent')} -> "
            f"{p.get('crps_challenger')} (rel {p.get('rel_delta'):+}, "
            f"P {p.get('p_improve')})")
    return [{
        "name": "Hierarchical Bayesian projection (Challenger A)",
        "scope": "prop projection distributions (line-free CRPS)",
        "verdict": "shipped" if gate.get("passed") else "rejected",
        "numbers": ("walk-forward 2021-2024 vs incumbent parametric families: "
                    + "; ".join(bits) + "; gate 0.90 + declared 1.5% relative"),
        "source": "book/bayes_projection_eval.json",
        "date": "2026-08-11",
    }]


def _seq_features() -> List[Dict]:
    book = _load("seq_features_eval.json")
    if not book:
        return []
    out = []
    for variant, label in (("b1", "GRU embeddings as GBDT features (B1)"),
                           ("b2", "GRU direct p_over head (B2)")):
        v = book.get(variant)
        if not isinstance(v, dict) or "gate" not in v:
            continue
        deltas = ", ".join(
            f"seed {k}: ll {s.get('ll_delta_pooled'):+} (P {s.get('p_improve_ll')})"
            for k, s in (v.get("seeds") or {}).items())
        out.append({
            "name": f"Sequence encoder — {label}",
            "scope": ("prop ranker features" if variant == "b1"
                      else "prop ranker (replacement)"),
            "verdict": "shipped" if v["gate"].get("passed") else "rejected",
            "numbers": f"walk-forward 2021-2024 A/B vs lean set: {deltas}; gate 0.90",
            "source": "book/seq_features_eval.json",
            "date": "2026-08-11",
        })
    verdict = book.get("verdict") or {}
    if verdict.get("stop_rule_tripped"):
        out.append({
            "name": "Props lever hunt — STOP RULE",
            "scope": "fablesfable_props track",
            "verdict": "research_only",
            "numbers": ("B1+B2 double rejection = third consecutive props "
                        "rejection (after pass_location 2026-07-30); "
                        "stop_after_consecutive_rejections=3 tripped — no new "
                        "props levers until new data or a genuinely different track"),
            "source": "book/seq_features_eval.json",
            "date": "2026-08-11",
        })
    return out


# Verdicts that predate the machine-written books; numbers live in the decision log.
STATIC_ENTRIES = [
    {
        "name": "O/U synthetic line: means -> median",
        "scope": "prop reference lines",
        "verdict": "rejected",
        "numbers": ("median barely moves the over-rate (rec_yds .360->.415, others "
                    "flat) and reintroduces the opposite bias; the synthetic split "
                    "is an artifact, not a calibration target"),
        "source": "docs/decisions_p3-5.md (2026-07-17)",
        "date": "2026-07-17",
    },
    {
        "name": "Isotonic/Platt calibration layers",
        "scope": "prop ranker probabilities",
        "verdict": "rejected",
        "numbers": "agent conflict + 2025-holdout inversions; raw GBDT probabilities kept",
        "source": "docs/decisions_p3-5.md / accuracy ledger (2026-07-16)",
        "date": "2026-07-16",
    },
    {
        "name": "Raw absence flags",
        "scope": "prop ranker features",
        "verdict": "rejected",
        "numbers": "priced in; player_depth_rank shipped instead",
        "source": "commit b7f4f49 (2026-07-16)",
        "date": "2026-07-16",
    },
    {
        "name": "Wilson-LB tier admission",
        "scope": "Top Bets bands",
        "verdict": "shipped",
        "numbers": "tiers admit on the 95% lower bound, never the point estimate",
        "source": "docs/decisions_p3-5.md (2026-07-17), PR #7/#8",
        "date": "2026-07-18",
    },
]


def collect() -> List[Dict]:
    entries = []
    entries.extend(_cover_calibration())
    entries.extend(_fair_value())
    entries.extend(_qb_haircut())
    entries.extend(_loc_features())
    entries.extend(_bayes_projection())
    entries.extend(_seq_features())
    entries.extend(STATIC_ENTRIES)
    entries.sort(key=lambda e: e.get("date", ""), reverse=True)
    return entries
