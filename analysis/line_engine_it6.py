#!/usr/bin/env python3
"""it6: combine the points-Elo rating edge (strong margin signal, fully priced)
with EPA pass/rush + QB/situational (weak margin signal, partially unpriced)
in one ridge model; it7: same but scored on DISAGREEMENT with the close
(the honest objective: only the residual vs market is monetizable)."""
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["LE_NO_MAIN"] = "1"

# reuse the data assembly from line_engine by importing its module namespace
import importlib.util
spec = importlib.util.spec_from_file_location(
    "le", os.path.join(os.path.dirname(os.path.abspath(__file__)), "line_engine.py"))

# --- instead of re-running line_engine's evaluations, rebuild the frame here
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "line_engine.py")).read()
     .split("# ---------------- iteration harness")[0])   # data assembly only

def feats6(d):
    return pd.DataFrame({
        "elo_edge": (d["off_home"] - d["def_away"]) - (d["off_away"] - d["def_home"]),
        "dp": (d["off_pass_h"] - d["def_pass_a"]) - (d["off_pass_a"] - d["def_pass_h"]),
        "dr": (d["off_rush_h"] - d["def_rush_a"]) - (d["off_rush_a"] - d["def_rush_h"]),
        "backup": d["backup_a"].astype(float) - d["backup_h"].astype(float),
        "rest_diff": d["rest_diff"],
        "short": d["short_a"].astype(float) - d["short_h"].astype(float),
        "tz": d["tz_shift_a"].astype(float),
        "hfa1": 1.0}, index=d.index)

def ridge(Xtr, ytr, Xte, lam=3.0):
    X, y = Xtr.to_numpy(), ytr.to_numpy()
    mu, sd = X.mean(0), X.std(0) + 1e-9; mu[-1], sd[-1] = 0, 1
    Xs = (X - mu) / sd
    b = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
    return ((Xte.to_numpy() - mu) / sd) @ b, Xs @ b, b

EVAL = [2020, 2021, 2022, 2023]
res_rows, coefs = [], []
for s in EVAL:
    tr, te = gdf[gdf["season"] < s], gdf[gdf["season"] == s]
    p, ptr, b = ridge(feats6(tr), tr["margin"], feats6(te))
    d = te.copy(); d["m_hat"] = p
    res_rows.append((d, tr, ptr)); coefs.append(b)

allte = pd.concat([r[0] for r in res_rows])
np_ = allte["margin"] != allte["spread_line"]
cover = allte["margin"] > allte["spread_line"]
pick = allte["m_hat"] > allte["spread_line"]
out = {"iteration": "it6_combined_ridge",
       "margin_MAE": round(float((allte["m_hat"] - allte["margin"]).abs().mean()), 3),
       "margin_corr": round(float(np.corrcoef(allte["m_hat"], allte["margin"])[0, 1]), 4),
       "SU_acc": round(float((np.sign(allte["m_hat"]) == np.sign(allte["margin"]))[allte["margin"] != 0].mean()), 4),
       "ATS_acc_all": round(float((pick == cover)[np_].mean()), 4),
       "coefs_mean(z)": dict(zip(feats6(gdf).columns, np.round(np.mean(coefs, 0), 3)))}

# kernel-EV selective (walk-forward error kernels)
H = N = 0; units = 0.0
for d, tr, ptr in res_rows:
    errs = (tr["margin"].to_numpy() - ptr).round().astype(int)
    vc = pd.Series(errs).value_counts(normalize=True).sort_index()
    ks, ps = vc.index.to_numpy(), vc.to_numpy()
    for r in d.itertuples():
        if r.margin == r.spread_line: continue
        need = r.spread_line - r.m_hat
        p_cover = ps[ks > need].sum() + 0.5 * ps[ks == round(need)].sum()
        p = max(p_cover, 1 - p_cover); ev = p * (10 / 11) - (1 - p)
        if ev < 0.035: continue
        win = ((p_cover >= 0.5) == (r.margin > r.spread_line))
        N += 1; H += win; units += (10 / 11) if win else -1.0
out["kernel_selective"] = {"n_bets": N, "ATS_acc": round(H / N, 4) if N else None,
                           "units_at_-110": round(units, 1)}

# disagreement bands
d = allte[np_].copy(); d["gap"] = d["m_hat"] - d["spread_line"]
d["hit"] = (d["gap"] > 0) == (d["margin"] > d["spread_line"])
out["disagreement_bands"] = {f"{lo}-{hi}pts": {"n": int(m.sum()), "ATS_acc": round(float(d.loc[m, "hit"].mean()), 4)}
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]
    if (m := d["gap"].abs().between(lo, hi, inclusive="left")).sum() >= 30}

print(json.dumps(out, indent=1))
book = json.load(open(os.path.join(ROOT, "book", "line_engine_iterations.json")))
book["iterations"].append(out)
json.dump(book, open(os.path.join(ROOT, "book", "line_engine_iterations.json"), "w"), indent=1)
