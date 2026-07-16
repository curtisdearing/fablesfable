#!/usr/bin/env python3
"""Accuracy harness: one command, one registry (accuracy loop plan, P2).

Collects the repo's CURRENT accuracy metrics from the canonical result
artifacts, pins the SHA-256 of every model input, and writes
data/accuracy_registry.json. The registry is the single scoreboard the
weekly lever loop reads and the accept gates are checked against.

    python3 analysis/eval_harness.py            # collect + write + print
    python3 analysis/eval_harness.py --check    # exit 1 if inputs drifted
                                                # since the last registry

This harness never computes new metrics itself: heavy evaluation stays in
the audited CLIs (ml_test.py, backtest.py, lean_backtest.py, tune_weights.py).
Rerun those first when a lever changes the model, then this collector.

Accept gates (pre-registered, accuracy_loop_plan.md): a lever is accepted
only if it moves a primary metric by at least the gate on the 2025 holdout,
evaluated once at accept time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUTS = [
    "historical/historical_pbp.parquet", "historical_lines.parquet",
    "historical/rosters_weekly.parquet", "historical/injuries.parquet",
    "historical/players_meta.parquet", "historical/ngs_receiving.parquet",
    "historical/contracts.parquet", "data/ml_frame.parquet",
    "config.json", "data/weights.json",
]

ACCEPT_GATES = {
    "ranker_log_loss": -0.002,
    "ranker_top5_hit_rate_pp": +0.5,
    "sim_brier": -0.002,
    "fantasy_mae_points": -0.05,   # tailstail-side gate, mirrored for reference
}


def sha256(path: str):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(path: str, default=None):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def collect() -> dict:
    ml = jload("data/ml_eval_results.json", {}) or {}
    bt = jload("data/backtest.json", {}) or {}
    lr = jload("data/lean_replay_2025.json", {}) or {}
    wt = jload("data/weight_tuning.json", {}) or {}
    le = jload("book/line_engine_iterations.json", {}) or {}

    seasons = {}
    for s, v in (ml.get("seasons") or {}).items():
        g = (v.get("models") or {}).get("gbdt") or {}
        seasons[s] = {
            "gbdt_log_loss": g.get("log_loss"), "gbdt_auc": g.get("auc"),
            "gbdt_top5_hit": (g.get("leans") or {}).get("hit_rate"),
            "gbdt_top1_hit": (g.get("leans") or {}).get("top1_hit_rate"),
            "composite_baseline_hit": (v.get("baseline_tuned_composite") or {}).get("hit_rate"),
        }

    metrics = {
        "prop_ranker_by_season": seasons,
        "prop_replay_2025": {
            "overall_hit": ((lr.get("leans") or {}).get("overall") or {}).get("hit_rate"),
            "top1_hit": ((lr.get("leans") or {}).get("top1_per_game") or {}).get("hit_rate"),
            "framing": "synthetic trailing-mean lines; breakeven proxy 0.5238 -- trend only, not money",
        },
        "game_sim_backtest": {
            "n_games": bt.get("n_games"), "brier": bt.get("brier"),
            "ats_pick_accuracy": bt.get("ats_pick_accuracy"),
            "corr_model": bt.get("corr_model"), "corr_market": bt.get("corr_market"),
        },
        "line_engine_iterations": {
            it.get("iteration"): {"ATS_all": it.get("ATS_acc_all"),
                                  "kernel_selective": it.get("kernel_selective")}
            for it in (le.get("iterations") or [])
        },
        "composite_ship_config": wt.get("ship_for_2026"),
        "ranker_feature_set": (jload("config.json", {}) or {}).get("ml_ranker", {}).get("features") and "lean" or "full",
    }
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any pinned input hash differs from the last registry")
    ap.add_argument("--output", default="data/accuracy_registry.json")
    args = ap.parse_args()

    inputs = {p: sha256(p) for p in INPUTS}
    out_path = os.path.join(ROOT, args.output)

    if args.check:
        prev = jload(args.output)
        if not prev:
            print("no previous registry -- nothing to check against")
            return 1
        drifted = {p: (prev.get("inputs", {}).get(p), h) for p, h in inputs.items()
                   if prev.get("inputs", {}).get(p) != h}
        if drifted:
            print("INPUT DRIFT since last registry:")
            for p, (old, new) in drifted.items():
                print(f"  {p}: {str(old)[:12]} -> {str(new)[:12]}")
            return 1
        print("inputs unchanged since last registry")
        return 0

    registry = {
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_head": git_head(),
        "holdout_policy": "tune on 2020-2024 walk-forward; 2025 evaluated once per lever at accept time",
        "accept_gates": ACCEPT_GATES,
        "inputs": inputs,
        "metrics": collect(),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(registry, fh, indent=1)

    m = registry["metrics"]
    print(f"accuracy registry @ {registry['git_head']} -> {args.output}")
    for s, v in sorted((m.get("prop_ranker_by_season") or {}).items()):
        print(f"  {s}: ranker top5 {v['gbdt_top5_hit']} top1 {v['gbdt_top1_hit']} "
              f"log_loss {v['gbdt_log_loss']} | composite {v['composite_baseline_hit']}")
    gb = m.get("game_sim_backtest") or {}
    print(f"  game sim: brier {gb.get('brier')} ATS {gb.get('ats_pick_accuracy')} "
          f"(market corr {gb.get('corr_market')})")
    print(f"  feature set: {m.get('ranker_feature_set')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
