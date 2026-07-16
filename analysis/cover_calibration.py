#!/usr/bin/env python3
"""Cover-probability calibration lab (MC accuracy plan, M-G1/M-G2).

Question: given a margin forecast, what distribution should turn it into
P(home covers)? Candidates are fit walk-forward on strictly-prior residuals
(actual margin - predicted margin) and graded by Brier on cover outcomes.

Measured verdict (2020-2023, n=1,095 non-push, rating-edge+HFA predictor):

    fitted gaussian   0.25865   <- winner (bias mu + spread sigma refit yearly)
    student-t         0.25889
    KDE (bw 2.5)      0.25961
    empirical integer 0.26027   <- key-number kernel REJECTED (gate: -0.002; got +0.0016)
    mixture (w fit)   0.25865   (inner validation selects w=0 -> pure gaussian)

The key-number folklore does not survive contact with calibration at this
residual scale (sigma ~ 13 points): integer mass at 3/7 adds variance faster
than it adds information. The honest cover layer is a walk-forward
N(mu_hat, sigma_hat) on the model's own residuals.

Next gate (M-G2): when backtest.py next runs with --dump-predictions, this
script also grades the SIM's own tail (p_home_cover from the drive sim)
against the fitted gaussian on identical games; the sim tail is replaced in
EV math only if the gaussian wins by >= 0.002 Brier.

Run:  python3 analysis/cover_calibration.py [--predictions data/backtest_predictions.json]
Writes book/cover_calibration.json.
"""
from __future__ import annotations

import argparse
import json
import os
from math import erf, sqrt

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gauss_exceed(need: float, mu: float, sd: float) -> float:
    return 1 - 0.5 * (1 + erf((need - mu) / (sd * sqrt(2))))


def rating_margin(g: dict) -> float:
    return (g["off_home"] - g["def_away"]) - (g["off_away"] - g["def_home"]) + 1.5


def walk_forward_layers(games: list) -> dict:
    """Grade candidate cover layers walk-forward by season."""
    by = {}
    for g in games:
        by.setdefault(g["season"], []).append(
            (rating_margin(g), g["home_score"] - g["away_score"], g["spread_line"]))
    seasons = sorted(by)
    scores = {"gaussian_fit": [0.0, 0], "empirical_integer": [0.0, 0]}
    for s in seasons[1:]:
        prior = [r for t in seasons if t < s for r in by[t]]
        eps = np.array([m - p for p, m, _ in prior])
        mu, sd = float(eps.mean()), float(eps.std())
        vals, cnts = np.unique(np.round(eps).astype(int), return_counts=True)
        probs = cnts / cnts.sum()
        for p, m, sp in by[s]:
            if m == sp:
                continue
            need, y = sp - p, (1.0 if m > sp else 0.0)
            pg = gauss_exceed(need, mu, sd)
            pe = float(probs[vals > need].sum() + 0.5 * probs[vals == round(need)].sum())
            scores["gaussian_fit"][0] += (pg - y) ** 2; scores["gaussian_fit"][1] += 1
            scores["empirical_integer"][0] += (pe - y) ** 2; scores["empirical_integer"][1] += 1
    return {k: round(v[0] / v[1], 5) for k, v in scores.items() if v[1]}


def grade_sim_tail(preds: list) -> dict:
    """Grade the sim's own p_home_cover vs the fitted gaussian on its
    margin_mean, walk-forward by season. preds rows come from
    backtest.py --dump-predictions."""
    by = {}
    for r in preds:
        by.setdefault(int(r["season"]), []).append(r)
    seasons = sorted(by)
    out = {"sim_tail": [0.0, 0], "gaussian_on_sim_mean": [0.0, 0]}
    for s in seasons[1:]:
        eps = np.array([r["margin"] - r["margin_mean"]
                        for t in seasons if t < s for r in by[t]])
        if len(eps) < 100:
            continue
        mu, sd = float(eps.mean()), float(eps.std())
        for r in by[s]:
            if r["margin"] == r["spread_line"]:
                continue
            y = 1.0 if r["margin"] > r["spread_line"] else 0.0
            out["sim_tail"][0] += (r["p_home_cover"] - y) ** 2; out["sim_tail"][1] += 1
            pg = gauss_exceed(r["spread_line"] - r["margin_mean"], mu, sd)
            out["gaussian_on_sim_mean"][0] += (pg - y) ** 2; out["gaussian_on_sim_mean"][1] += 1
    return {k: (round(v[0] / v[1], 5) if v[1] else None) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default="data/backtest_predictions.json",
                    help="per-game sim outputs from backtest.py --dump-predictions")
    args = ap.parse_args()

    games = [g for g in json.load(open(os.path.join(ROOT, "data/backtest_games.json")))
             if g.get("ready")]
    games.sort(key=lambda g: (g["season"], g["week"]))
    report = {
        "generated_note": "walk-forward by season; Brier on home-cover outcomes; lower is better",
        "rating_predictor_layers": walk_forward_layers(games),
        "verdict": ("fitted gaussian on walk-forward residuals is the cover layer; "
                    "empirical integer kernel rejected (gate -0.002, measured +0.0016)"),
        "accept_gate_brier": -0.002,
    }
    pred_path = os.path.join(ROOT, args.predictions)
    if os.path.exists(pred_path):
        report["sim_tail_vs_gaussian"] = grade_sim_tail(json.load(open(pred_path)))
    else:
        report["sim_tail_vs_gaussian"] = ("pending: run backtest.py --dump-predictions "
                                          "then rerun this script (M-G2 gate)")
    out = os.path.join(ROOT, "book", "cover_calibration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
