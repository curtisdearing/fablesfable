#!/usr/bin/env python3
"""Evaluate analysis/role_opportunity_protocol.json (frozen before this script existed).

DISCOVERY evidence only: 2021-2025 and 2026 W1-2 are all previously exposed.

    python analysis/role_opportunity_eval.py --hist <copy of pinned historical/> --out <dir>

Reads only the supplied directory (copy it first; nothing is written there),
writes rows/metrics to --out and the public summary to
analysis/role_opportunity_results.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import special, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import prop_backtest  # noqa: E402
from nflvalue import features, football_forecast as ff, role_opportunity as ro  # noqa: E402
from nflvalue.projection import MARKETS  # noqa: E402

PROTOCOL = os.path.join(ROOT, "analysis", "role_opportunity_protocol.json")
EVAL_SEASONS = (2021, 2022, 2023, 2024, 2025)
EXPOSED_2026 = (2026,)
#: C_NC is a POST-HOC diagnostic (added after the v1 gate result): C without
#: team-share conservation.  It is reported, never gated.
ARMS = ("INC", "PRIOR", "CURR", "C", "C2", "EP", "EC", "C_NC")
ENDPOINTS = {  # endpoint -> (market spec key or None, roles, actual col, volume, efficiency quantity)
    "targets": (None, ("WR", "TE"), "targets", "targets", None),
    "receptions": ("receptions", ("WR", "TE"), "receptions", "targets", "catch_rate"),
    "receiving_yards": ("receiving_yards", ("WR", "TE"), "rec_yards", "targets", "ypt"),
    "rush_attempts": ("rush_attempts", ("RB",), "carries", "carries", None),
    "rushing_yards": ("rushing_yards", ("RB",), "rush_yards", "carries", "ypc"),
    "pass_attempts": ("pass_attempts", ("QB",), "pass_attempts", "pass_attempts", None),
    "passing_yards": ("passing_yards", ("QB",), "pass_yards", "pass_attempts", "ypa"),
}
VOL_KEY = {"targets": ("target_share", "team_pass_att", "expected_targets"),
           "carries": ("carry_share", "team_rush_att", "expected_carries"),
           "pass_attempts": ("pass_share", "team_pass_att", "expected_pass_attempts")}
SEED = 20260922


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(hist):
    frames = [pd.read_parquet(os.path.join(hist, "historical_pbp.parquet"), columns=features.PBP_COLUMNS)]
    for s in (2024, 2025, 2026):
        frames.append(pd.read_parquet(os.path.join(hist, f"pbp_{s}.parquet"), columns=features.PBP_COLUMNS))
    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp[pbp["season_type"] == "REG"].reset_index(drop=True)
    rosters = pd.read_parquet(os.path.join(hist, "rosters_weekly.parquet"))
    return pbp, rosters


# ------------------------------------------------------------------ scoring --
def crps(y, mean, sd, dist):
    y, mean, sd = (np.asarray(a, float) for a in (y, mean, sd))
    mean, sd = np.maximum(mean, 1e-6), np.maximum(sd, 1e-6)
    if dist == "normal":
        z = (y - mean) / sd
        return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    if dist == "gamma":  # exact (Scheuerer & Moller 2015), rate b
        a, b = (mean / sd) ** 2, mean / sd ** 2
        return (y * (2 * stats.gamma.cdf(y, a, scale=1 / b) - 1)
                - a / b * (2 * stats.gamma.cdf(y, a + 1, scale=1 / b) - 1) - 1 / (b * special.beta(a, 0.5)))
    if dist == "negbinom":  # exact discrete sum
        var = np.maximum(sd ** 2, mean * 1.01)
        n, p = mean ** 2 / (var - mean), mean / var
        kmax = int(max(np.nanmax(y), np.nanmax(stats.nbinom.ppf(0.99999, n, p)))) + 2
        k = np.arange(kmax + 1)[None, :]
        F = stats.nbinom.cdf(k, n[:, None], p[:, None])
        return ((F - (y[:, None] <= k)) ** 2).sum(axis=1)
    raise ValueError(dist)


def interval(mean, sd, dist, lo=0.1, hi=0.9):
    mean, sd = np.maximum(np.asarray(mean, float), 1e-6), np.maximum(np.asarray(sd, float), 1e-6)
    if dist == "normal":
        return stats.norm.ppf(lo, mean, sd), stats.norm.ppf(hi, mean, sd)
    if dist == "gamma":
        a, sc = (mean / sd) ** 2, sd ** 2 / mean
        return stats.gamma.ppf(lo, a, scale=sc), stats.gamma.ppf(hi, a, scale=sc)
    var = np.maximum(sd ** 2, mean * 1.01)
    n, p = mean ** 2 / (var - mean), mean / var
    return stats.nbinom.ppf(lo, n, p), stats.nbinom.ppf(hi, n, p)


def boot_delta(df, col_a, col_b, reps=1000):
    """95% game-cluster CI of mean(col_a) - mean(col_b)."""
    g = df.groupby("game_id").agg(a=(col_a, "sum"), b=(col_b, "sum"), n=(col_a, "size"))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(g), size=(reps, len(g)))
    a, b, n = g["a"].to_numpy(), g["b"].to_numpy(), g["n"].to_numpy()
    d = (a[idx].sum(1) - b[idx].sum(1)) / n[idx].sum(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ------------------------------------------------------------------ arms -----
def challenger_rows(games, pw, seasons):
    out, hypers = [], {}
    for S in seasons:
        hyper = ro.fit_hyperparameters(games, before_season=S)
        hypers[S] = hyper
        for W in sorted(pw.loc[pw["season"] == S, "week"].unique()):
            wk = pw[(pw["season"] == S) & (pw["week"] == W) & pw["role"].isin(["QB", "RB", "WR", "TE"])]
            tg = pd.DataFrame({"player_id": wk["player_id"], "team": wk["team"],
                               "position": wk["role"], "game_id": wk["game_id"]})
            as_of = dt.datetime(S, 9, 1, 12, tzinfo=dt.timezone.utc) + dt.timedelta(days=7 * (int(W) - 1))
            res = ro.forecast_opportunity(games, tg, season=S, week=int(W), as_of=as_of, hyper=hyper)
            for p in res["players"]:
                rec = {"season": S, "week": int(W), "player_id": p["player_id"], "regime": p["regime"]}
                for vol, (sq, tq, name) in VOL_KEY.items():
                    s = p.get(sq)
                    if s is None:
                        continue
                    t = res["teams"][p["team"]][tq]
                    cur_s = s["current_estimate"] if s["current_estimate"] is not None else s["prior_estimate"]
                    cur_t = t["current_estimate"] if t["current_estimate"] is not None else t["prior_estimate"]
                    rec.update({f"C_{vol}": p[name], f"PRIOR_{vol}": p[f"{name}_prior_only"],
                            f"CNC_{vol}": t["posterior"] * s["posterior"],
                                f"CURR_{vol}": cur_s * cur_t, f"ng_{vol}": s["current_games"],
                                f"w_{vol}": s["w_current"], f"wteam_{vol}": t["w_current"]})
                for eq in ro.EFFICIENCY_QUANTITIES:
                    e = p.get(eq)
                    if e is None:
                        continue
                    rec.update({f"post_{eq}": e["posterior"], f"prior_{eq}": e["prior_estimate"],
                                f"cur_{eq}": e["current_estimate"] if e["current_estimate"] is not None
                                else e["prior_estimate"], f"w_{eq}": e["w_current"]})
                out.append(rec)
    return pd.DataFrame(out), hypers


def build_frame(pw, opd, tw, ch, disp):
    team_idx = {(r.season, r.week, r.team): r._asdict() for r in tw.itertuples(index=False)}
    opp_idx = {(r.season, r.week, r.defteam, r.role): r._asdict() for r in opd.itertuples(index=False)}
    frames = {}
    for ep, (mkt, roles, actual, vol, eq) in ENDPOINTS.items():
        base_mkt = mkt or "receptions"
        rows = pw[pw["role"].isin(roles) & pw["season"].isin(EVAL_SEASONS + EXPOSED_2026)]
        pr = prop_backtest._predictions_for_market(rows, base_mkt, team_idx, opp_idx)
        pr = pr.merge(ch, on=["season", "week", "player_id"], how="left")
        pr["actual"] = pr[actual]
        inc_vol = pr["volume_pred"]
        inc_eff = pr["efficiency_pred"] if eq else 1.0
        opp = pr["opp_factor"] if mkt and MARKETS[mkt]["use_opp_factor"] else 1.0
        e_post = pr[f"post_{eq}"] if eq else 1.0
        e_prior = pr[f"prior_{eq}"] if eq else 1.0
        e_cur = pr[f"cur_{eq}"] if eq else 1.0
        vols = {"INC": inc_vol, "PRIOR": pr[f"PRIOR_{vol}"], "CURR": pr[f"CURR_{vol}"], "C": pr[f"C_{vol}"],
                "C2": pr[f"C_{vol}"], "EP": pr[f"C_{vol}"], "EC": pr[f"C_{vol}"], "C_NC": pr[f"CNC_{vol}"]}
        effs = {"INC": inc_eff, "PRIOR": inc_eff, "CURR": inc_eff, "C": inc_eff, "C_NC": inc_eff,
                "C2": e_post, "EP": e_prior, "EC": e_cur}
        for a in ARMS:
            pr[f"m_{a}"] = np.maximum(np.asarray(vols[a] * effs[a] * (opp if mkt else 1.0), float), 0.0)
        pr["m_INC"] = pr["mean_pred"] if mkt else inc_vol
        pr["ng"] = pr[f"ng_{vol}"]
        pr["w_share"] = pr[f"w_{vol}"]
        pr["w_eff"] = pr[f"w_{eq}"] if eq else np.nan
        ok = np.isfinite(pr[[f"m_{a}" for a in ARMS] + ["actual"]].to_numpy(float)).all(1)
        dropped = int((~ok).sum())
        pr = pr[ok].copy()
        dist = MARKETS[mkt]["dist"] if mkt else None
        if mkt:
            for a in ARMS:
                sd = np.array([ff.conditional_sd(mkt, float(m), disp) or np.nan for m in pr[f"m_{a}"]])
                sd = np.where(np.isfinite(sd), sd, disp["markets"][mkt]["sd_floor"])
                pr[f"crps_{a}"] = crps(pr["actual"].to_numpy(), pr[f"m_{a}"].to_numpy(), sd, dist)
                lo, hi = interval(pr[f"m_{a}"].to_numpy(), sd, dist)
                y = pr["actual"].to_numpy()
                pr[f"cov_{a}"] = ((y >= lo) & (y <= hi)).astype(float)
        for a in ARMS:
            pr[f"ae_{a}"] = (pr["actual"] - pr[f"m_{a}"]).abs()
            pr[f"err_{a}"] = pr["actual"] - pr[f"m_{a}"]
        pr.attrs["dropped"] = dropped
        frames[ep] = pr
    return frames


def summarize(df, has_dist):
    out = {"n_rows": int(len(df)), "n_games": int(df["game_id"].nunique()),
           "n_players": int(df["player_id"].nunique()), "mean_actual": float(df["actual"].mean()), "arms": {}}
    for a in ARMS:
        r = {"mae": float(df[f"ae_{a}"].mean()), "bias": float(df[f"err_{a}"].mean()),
             "mean_forecast": float(df[f"m_{a}"].mean())}
        if has_dist:
            r["crps"] = float(df[f"crps_{a}"].mean())
            r["cov80"] = float(df[f"cov_{a}"].mean())
        if a != "INC":
            r["d_mae_vs_inc"] = r["mae"] - float(df["ae_INC"].mean())
            r["d_mae_ci"] = boot_delta(df, f"ae_{a}", "ae_INC")
            if has_dist:
                r["d_crps_vs_inc"] = r["crps"] - float(df["crps_INC"].mean())
                r["d_crps_ci"] = boot_delta(df, f"crps_{a}", "crps_INC")
        out["arms"][a] = r
    return out


def bucket(n):
    n = int(n)
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else "9+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    proto_sha = sha(PROTOCOL)
    inputs = {f: sha(os.path.join(args.hist, f)) for f in sorted(os.listdir(args.hist)) if f.endswith(".parquet")}
    pbp, rosters = load(args.hist)
    pw = features.build_player_week(pbp, rosters=rosters)
    opd = features.build_opp_pos_def(pbp, rosters=rosters)
    tw = features.build_team_week(pbp)
    games = ro.player_games_from_player_week(pw)
    pw = pw.assign(game_id=games["game_id"].to_numpy())
    disp = ff.load_dispersion()
    ch, hypers = challenger_rows(games, pw, EVAL_SEASONS + EXPOSED_2026)
    frames = build_frame(pw, opd, tw, ch, disp)

    res = {"protocol_sha256": proto_sha, "inputs_sha256": inputs, "evidence_status": "DISCOVERY (exposed seasons)",
           "crps_method_note": "closed-form exact CRPS (normal, gamma, discrete negbinom sum) instead of the "
                               "protocol's 400-draw Monte Carlo: same estimand, no MC noise; decided before results",
           "hyperparameters": {str(k): v for k, v in hypers.items()}, "pooled": {}, "by_season": {},
           "by_current_games": {}, "by_regime": {}, "exposed_2026": {}, "dropped_nonfinite": {}}
    for ep, df in frames.items():
        has_dist = ENDPOINTS[ep][0] is not None
        df.to_csv(os.path.join(args.out, f"rows_{ep}.csv.gz"), index=False)
        ev = df[df["season"].isin(EVAL_SEASONS)]
        res["dropped_nonfinite"][ep] = df.attrs.get("dropped", 0)
        res["pooled"][ep] = summarize(ev, has_dist)
        res["by_season"][ep] = {str(s): summarize(ev[ev["season"] == s], has_dist) for s in EVAL_SEASONS}
        ev = ev.assign(bk=ev["ng"].map(bucket))
        res["by_current_games"][ep] = {b: dict(summarize(x, has_dist), mean_w_share=float(x["w_share"].mean()),
                                               mean_w_eff=float(x["w_eff"].mean()) if ENDPOINTS[ep][4] else None)
                                       for b, x in ev.groupby("bk")}
        res["by_regime"][ep] = {r: summarize(x, has_dist) for r, x in ev.groupby("regime")}
        e26 = df[df["season"] == 2026]
        res["exposed_2026"][ep] = {str(w): summarize(x, has_dist) for w, x in e26.groupby("week")}

    p = res["pooled"]
    opp_eps = ("targets", "rush_attempts", "pass_attempts")
    mkts = ("receiving_yards", "receptions", "rushing_yards", "passing_yards", "pass_attempts", "rush_attempts")
    g1 = {e: p[e]["arms"]["C"]["d_mae_ci"][1] < 0 for e in opp_eps}
    g2 = {e: abs(p[e]["arms"]["C"]["bias"]) <= abs(p[e]["arms"]["INC"]["bias"]) + 0.01 * p[e]["mean_actual"]
          for e in mkts + ("targets",)}
    g3_n = sum(p[m]["arms"]["C"]["d_crps_vs_inc"] <= 0 for m in mkts)
    g3_worse = [m for m in mkts if p[m]["arms"]["C"]["d_crps_ci"][0] > 0]
    g4 = {m: p[m]["arms"]["C"]["cov80"] >= p[m]["arms"]["INC"]["cov80"] - 0.01 for m in mkts}
    passed = all(g1.values()) and all(g2.values()) and g3_n >= 4 and not g3_worse and all(g4.values())
    res["gate"] = {"G1_mae_ci_upper_lt0": g1, "G2_bias": g2, "G3_crps_nonworse_count": int(g3_n),
                   "G3_markets_significantly_worse": g3_worse, "G4_cov80": g4, "passed": bool(passed),
                   "decision": ("discovery pass: recommend deployed shadow + prospective 2026 W3+ test" if passed
                                else "gate failed: deployed shadow / display-only diagnostic")}
    res["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "role_opportunity_results_full.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    public = {k: v for k, v in res.items() if k != "by_regime"} | {"by_regime": res["by_regime"]}
    with open(os.path.join(ROOT, "analysis", "role_opportunity_results.json"), "w") as f:
        json.dump(public, f, indent=1, default=float)
    print(json.dumps(res["gate"], indent=1))
    for ep in ENDPOINTS:
        a = p[ep]["arms"]
        print(ep, p[ep]["n_rows"], {k: (round(v["mae"], 3), round(v["bias"], 3), round(v.get("crps", np.nan), 3))
                                    for k, v in a.items()})


if __name__ == "__main__":
    main()
