#!/usr/bin/env python3
"""Evaluate analysis/football_only_protocol.json (frozen before the test window was opened).

Run from a repository root whose historical/ cache is present:
    python analysis/football_only_eval.py [--real-lines rows.csv.gz] [--out analysis/football_only_results.json]
Writes the fitted dispersion to data/dispersion_v1.json only with --write-dispersion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import prop_backtest  # noqa: E402
from nflvalue import football_forecast as ff  # noqa: E402
from nflvalue import projection  # noqa: E402
from nflvalue.candidates import build_week_inputs, synthetic_lines  # noqa: E402

PROTOCOL = os.path.join(ROOT, "analysis", "football_only_protocol.json")
MARKETS6 = ["receiving_yards", "receptions", "rushing_yards", "passing_yards",
            "pass_attempts", "rush_attempts"]
CAL, TEST = (2023, 2024), (2025,)


def sf(x, mean, sd, dist):
    """Vectorized projection._SF (same parameterizations)."""
    mean = np.maximum(np.asarray(mean, float), 1e-6)
    sd = np.maximum(np.asarray(sd, float), 1e-6)
    if dist == "normal":
        return stats.norm.sf(x, loc=mean, scale=sd)
    if dist == "gamma":
        return stats.gamma.sf(x, a=(mean / sd) ** 2, scale=sd ** 2 / mean)
    if dist == "negbinom":
        var = np.maximum(sd ** 2, mean * 1.01)
        return stats.nbinom.sf(np.floor(x), mean ** 2 / (var - mean), mean / var)
    raise ValueError(dist)


def mid_pit(y, mean, sd, dist):
    if dist == "negbinom":
        return 1 - 0.5 * (sf(y, mean, sd, dist) + sf(y - 1, mean, sd, dist))
    return 1 - sf(y, mean, sd, dist)


def cluster_boot(df, stat, reps=1000, seed=20260922):
    """95% CI of stat(df) resampling whole games."""
    rng = np.random.default_rng(seed)
    games = df["game_id"].unique()
    idx = {g: ix for g, ix in df.groupby("game_id").indices.items()}
    vals = []
    for _ in range(reps):
        pick = rng.choice(games, len(games), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        vals.append(stat(df.iloc[rows]))
    return [round(float(np.percentile(vals, 2.5)), 5), round(float(np.percentile(vals, 97.5)), 5)]


def game_id_col(pw):
    """One id per GAME (both teams' rows share it): season_week_teamA_teamB, sorted pair.

    The protocol clusters by game; an earlier draft keyed on (team, defteam)
    order and so split each game into two clusters.
    """
    s = pw["season"].astype(int).astype(str)
    w = pw["week"].astype(int).map("{:02d}".format)
    pair = [("_".join(sorted((a, b)))) for a, b in zip(pw["team"], pw["defteam"])]
    return s + "_" + w + "_" + pd.Series(pair, index=pw.index)


def margins_all(schedules, seasons):
    out = {}
    for (se, wk) in schedules[schedules["season"].isin(seasons)][["season", "week"]].drop_duplicates().itertuples(index=False):
        for team, m in ff.football_margins(schedules, int(se), int(wk)).items():
            out[(int(se), int(wk), team)] = m
    return out


def spread_margins(schedules, seasons):
    out = {}
    for g in schedules[schedules["season"].isin(seasons)].itertuples(index=False):
        if pd.notna(g.spread_line):
            out[(int(g.season), int(g.week), g.home_team)] = float(g.spread_line)
            out[(int(g.season), int(g.week), g.away_team)] = -float(g.spread_line)
    return out


def arm_means(rows, market, inputs, margin_idx):
    spec = projection.MARKETS[market]
    means = []
    for r in rows.itertuples(index=False):
        pr = r._asdict()
        team_row = inputs.team_idx.get((r.season, r.week, r.team))
        opp_row = inputs.opp_idx.get((r.season, r.week, r.defteam, r.role)) if spec["use_opp_factor"] else None
        m = None if margin_idx is None else margin_idx.get((int(r.season), int(r.week), r.team))
        gs = projection.game_script_multipliers(m)
        means.append(projection.project(pr, market, team_row=team_row, opp_row=opp_row,
                                        line=None, sd=1.0, game_script=gs)["mean"])
    return np.asarray(means, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-lines", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "football_only_results.json"))
    ap.add_argument("--write-dispersion", action="store_true")
    a = ap.parse_args()
    proto_sha = hashlib.sha256(open(PROTOCOL, "rb").read()).hexdigest()
    inputs = build_week_inputs()
    pw, sched = inputs.pw, inputs.schedules
    seasons = CAL + TEST
    m_foot = margins_all(sched, seasons)
    m_spread = spread_margins(sched, seasons)
    res = {"protocol_sha256": proto_sha, "forecast_version": ff.FORECAST_VERSION,
           "margin": {}, "dispersion": {}, "real_line_discovery": None}
    disp_params = {"version": "dispersion_v1", "fit_seasons": list(CAL), "protocol_sha256": proto_sha,
                   "markets": {}}
    margin_rows = []
    for market in MARKETS6 + ["anytime_td"]:
        spec = projection.MARKETS[market]
        base = prop_backtest._predictions_for_market(pw[pw["season"] <= max(seasons)], market,
                                                     inputs.team_idx, inputs.opp_idx)
        base = base[base["eligible_for_shortlist"] & base["actual"].notna()]
        hist = base[base["season"] < TEST[0]]
        d0_sd = float((hist["actual"] - hist["mean_pred"]).std(ddof=1))
        rows = base[base["season"].isin(seasons)].copy()
        rows["game_id"] = game_id_col(rows)
        rows["A1"] = arm_means(rows, market, inputs, None)
        rows["A2"] = arm_means(rows, market, inputs, m_foot)
        rows["A0"] = arm_means(rows, market, inputs, m_spread)
        rows["market"] = market
        margin_rows.append(rows[["season", "game_id", "market", "actual", "A0", "A1", "A2"]])
        if market not in MARKETS6:
            continue
        cal = rows[rows["season"].isin(CAL)]
        p = ff.fit_conditional_sd(cal["A1"].to_numpy(), cal["actual"].to_numpy())
        disp_params["markets"][market] = p
        test = rows[rows["season"].isin(TEST)].copy()
        synth = synthetic_lines(inputs, market)
        test["line"] = synth.reindex(test.index).to_numpy()
        test = test[test["line"].notna()]
        test["sd0"] = d0_sd
        test["sd1"] = [ff.conditional_sd(market, m, disp_params) or d0_sd for m in test["A1"]]
        test["y_over"] = (test["actual"] > test["line"]).astype(float)
        out = {"n_rows": int(len(test)), "n_games": int(test["game_id"].nunique()),
               "d0_sd": round(d0_sd, 3), "d1_params": p}
        for arm, sdc in (("D0", "sd0"), ("D1", "sd1")):
            pit = mid_pit(test["actual"].to_numpy(), test["A1"].to_numpy(), test[sdc].to_numpy(), spec["dist"])
            pov = np.clip(sf(test["line"].to_numpy(), test["A1"].to_numpy(), test[sdc].to_numpy(), spec["dist"]), 1e-6, 1 - 1e-6)
            test["p_" + arm] = pov
            out[arm] = {"cov80": round(float(np.mean((pit >= .1) & (pit <= .9))), 4),
                        "cov50": round(float(np.mean((pit >= .25) & (pit <= .75))), 4),
                        "brier": round(float(np.mean((pov - test["y_over"]) ** 2)), 5),
                        "logloss": round(float(-np.mean(test["y_over"] * np.log(pov) + (1 - test["y_over"]) * np.log(1 - pov))), 5),
                        "mean_abs_p_minus_half": round(float(np.mean(np.abs(pov - .5))), 4)}
        out["over_rate"] = round(float(test["y_over"].mean()), 4)
        dB = lambda d: float(np.mean((d["p_D1"] - d["y_over"]) ** 2 - (d["p_D0"] - d["y_over"]) ** 2))
        out["dBrier_D1_minus_D0"] = round(dB(test), 5)
        out["dBrier_CI"] = cluster_boot(test.reset_index(drop=True), dB)
        passes = (abs(out["D1"]["cov80"] - .8) < abs(out["D0"]["cov80"] - .8)) and out["dBrier_CI"][1] <= 0
        out["decision"] = "D1_primary" if passes else "D0_primary_D1_shadow"
        res["dispersion"][market] = out
        print(market, json.dumps(out)[:400], flush=True)

    mr = pd.concat(margin_rows, ignore_index=True)
    for window, ss in (("calibration", CAL), ("test", TEST)):
        w = mr[mr["season"].isin(ss)].reset_index(drop=True)
        block = {"n_rows": int(len(w)), "n_games": int(w["game_id"].nunique()), "markets": {}}
        for arm in ("A0", "A1", "A2"):
            block[f"mae_{arm}_pooled_scaled"] = None
        for market, g in w.groupby("market"):
            g = g.reset_index(drop=True)
            mk = {arm: round(float(np.mean(np.abs(g["actual"] - g[arm]))), 4) for arm in ("A0", "A1", "A2")}
            mk["n"] = int(len(g))
            if window == "test":
                d21 = lambda d: float(np.mean(np.abs(d["actual"] - d["A2"]) - np.abs(d["actual"] - d["A1"])))
                d01 = lambda d: float(np.mean(np.abs(d["actual"] - d["A0"]) - np.abs(d["actual"] - d["A1"])))
                mk["dMAE_A2_A1"], mk["dMAE_A2_A1_CI"] = round(d21(g), 5), cluster_boot(g, d21, reps=500)
                mk["dMAE_A0_A1"], mk["dMAE_A0_A1_CI"] = round(d01(g), 5), cluster_boot(g, d01, reps=500)
            block["markets"][market] = mk
        # pooled on a scale-free basis: each market's |error| divided by its A1 MAE
        scale = w.groupby("market").apply(lambda g: np.mean(np.abs(g["actual"] - g["A1"])))
        w["s"] = w["market"].map(scale)
        for arm in ("A0", "A1", "A2"):
            block[f"mae_{arm}_pooled_scaled"] = round(float(np.mean(np.abs(w["actual"] - w[arm]) / w["s"])), 5)
        if window == "test":
            dp = lambda d: float(np.mean((np.abs(d["actual"] - d["A2"]) - np.abs(d["actual"] - d["A1"])) / d["s"]))
            block["dMAE_A2_A1_pooled_scaled"] = round(dp(w), 6)
            block["dMAE_A2_A1_pooled_CI"] = cluster_boot(w, dp, reps=500)
            ok = block["dMAE_A2_A1_pooled_CI"][1] < 0 and all(
                m["dMAE_A2_A1_CI"][0] <= 0 for m in block["markets"].values())
            block["decision"] = "A2_football_primary" if ok else "A1_neutral_primary_A2_shadow"
        res["margin"][window] = block
    print("margin", json.dumps(res["margin"]["test"])[:1500], flush=True)

    if a.real_lines:
        d = pd.read_csv(a.real_lines)
        d = d[d["is_latest"] & d["is_unique_event"] & d["market"].isin(MARKETS6)
              & d["y_over"].notna() & d["mean_new"].notna()].copy()
        d["p_D1"] = [projection.p_over(m, ff.conditional_sd(mk, m, disp_params) or s, pt, dist)
                     for m, s, pt, dist, mk in zip(d["mean_new"], d["sd_new"], d["point"], d["dist_new"], d["market"])]
        from sklearn.metrics import roc_auc_score
        rl = {"note": "DISCOVERY ONLY (2026 wk1-2, already examined); not used by any rule",
              "n_events": int(len(d)), "n_games": int(d["game_id"].nunique()), "arms": {}}
        for arm, col in (("D0_incumbent", "p_new"), ("D1_conditional", "p_D1"), ("market_consensus", "consensus_p_over")):
            q = d[[col, "y_over"]].dropna()
            p = q[col].clip(1e-6, 1 - 1e-6)
            rl["arms"][arm] = {"n": int(len(q)), "brier": round(float(np.mean((p - q["y_over"]) ** 2)), 4),
                               "logloss": round(float(-np.mean(q["y_over"] * np.log(p) + (1 - q["y_over"]) * np.log(1 - p))), 4),
                               "auc": round(float(roc_auc_score(q["y_over"], p)), 4),
                               "mean_abs_p_minus_half": round(float(np.mean(np.abs(p - .5))), 4)}
        res["real_line_discovery"] = rl
        print("real-line", json.dumps(rl), flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    if a.write_dispersion:
        with open(ff.DISPERSION_PATH, "w") as f:
            json.dump(disp_params, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
