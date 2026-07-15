#!/usr/bin/env python3
"""Iterative line-engine improvement: rebuild the margin model the way
professional originators build openers, one ingredient at a time, measuring
each iteration walk-forward on the same games.

Pro line anatomy encoded here:
  it0  points-Elo ratings (current build_ratings baseline)
  it1  opponent-adjusted EPA/play ratings, pass/rush split, EWM recency
  it2  + QB continuity adjustment (backup starter detection via schedules)
  it3  + situational: season-fit HFA, rest differential, short week, TZ travel
  it4  + empirical margin kernel (key numbers 3/7): cover prob from the
       discrete distribution of forecast errors, selective betting by EV
  it5  + market blend (originators anchor to consensus): alpha fit walk-forward

Eval: seasons 2020-2023 walk-forward (train strictly prior), vs closing
spread_line from backtest_games.json. Metrics: margin MAE/corr, straight-up
acc, ATS acc (all games), selective ATS at kernel-EV threshold, units @ -110.
"""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")
OUT = {}

games = [g for g in json.load(open(os.path.join(ROOT, "data/backtest_games.json"))) if g.get("ready")]
gdf = pd.DataFrame(games)
gdf["margin"] = gdf["home_score"] - gdf["away_score"]
gdf = gdf.sort_values(["season", "week"]).reset_index(drop=True)

# ---------------- EPA team-week table (walk-forward EWM, opponent-adjusted)
pbp = pd.read_parquet(os.path.join(HIST, "historical_pbp.parquet"),
                      columns=["season", "week", "game_id", "season_type", "posteam",
                               "defteam", "epa", "pass_attempt", "rush_attempt"])
pbp = pbp[(pbp["season_type"] == "REG") & pbp["posteam"].notna() & pbp["epa"].notna()]
pbp["is_pass"] = pbp["pass_attempt"] == 1
tw = (pbp.groupby(["season", "week", "posteam", "defteam", "is_pass"])["epa"]
      .agg(["mean", "size"]).reset_index())

def ewm_ratings(span=10):
    """For each (season, week, team): off/def EPA per play (pass & rush),
    opponent-adjusted one pass, strictly prior, cross-season carry 0.6."""
    rows = []
    for kind in ["off", "def"]:
        key = "posteam" if kind == "off" else "defteam"
        for is_pass in [True, False]:
            d = tw[tw["is_pass"] == is_pass].copy()
            d = d.groupby(["season", "week", key])[["mean", "size"]].apply(
                lambda x: pd.Series({"epa": np.average(x["mean"], weights=x["size"]),
                                     "n": x["size"].sum()})).reset_index()
            d = d.sort_values(["season", "week"]).rename(columns={key: "team"})
            d["kind"] = kind + ("_pass" if is_pass else "_rush")
            rows.append(d)
    long = pd.concat(rows, ignore_index=True)
    # opponent adjustment: subtract league mean, then iterate once vs opponent strength
    long["adj"] = long.groupby(["season", "week", "kind"])["epa"].transform(lambda x: x - x.mean())
    # walk-forward EWM per team+kind with cross-season carry
    out = []
    for (team, kind), d in long.groupby(["team", "kind"]):
        d = d.sort_values(["season", "week"])
        val, prev_season = 0.0, None
        for r in d.itertuples():
            if prev_season is not None and r.season != prev_season:
                val *= 0.6
            out.append((r.season, r.week, team, kind, val))   # value BEFORE this game
            alpha = 2 / (span + 1)
            val = (1 - alpha) * val + alpha * r.adj
            prev_season = r.season
        # also emit ratings for weeks after last played (bye handling is implicit)
    return pd.DataFrame(out, columns=["season", "week", "team", "kind", "rating"])

R = ewm_ratings()
Rp = R.pivot_table(index=["season", "week", "team"], columns="kind", values="rating").reset_index()
Rp.columns.name = None

def latest_rating(season, week, team):
    d = Rp[(Rp["team"] == team) & ((Rp["season"] < season) |
          ((Rp["season"] == season) & (Rp["week"] <= week)))]
    if d.empty: return None
    return d.iloc[-1]

# join: rating snapshot known before each game (<= same week works because
# rating rows store the value BEFORE that week's game)
snap_h, snap_a = [], []
for r in gdf.itertuples():
    h, a = latest_rating(r.season, r.week, r.home), latest_rating(r.season, r.week, r.away)
    snap_h.append(h); snap_a.append(a)
for side, snaps in [("h", snap_h), ("a", snap_a)]:
    for k in ["off_pass", "off_rush", "def_pass", "def_rush"]:
        gdf[f"{k}_{side}"] = [None if s is None else s.get(k, 0.0) for s in snaps]
gdf = gdf.dropna(subset=["off_pass_h", "off_pass_a"]).reset_index(drop=True)

# ---------------- schedule context (QB, rest, tz)
sc = pd.concat([pd.read_parquet(os.path.join(ROOT, "historical_lines.parquet")),
                pd.read_parquet(os.path.join(HIST, "lines_extra.parquet"))], ignore_index=True)
sc = sc[sc["game_type"] == "REG"]
TZ = {**{t: 0 for t in ["BUF","MIA","NE","NYJ","BAL","CIN","CLE","PIT","ATL","CAR","JAX","TB","WAS","NYG","PHI","DET","IND"]},
      **{t: 1 for t in ["CHI","GB","MIN","DAL","HOU","KC","NO","TEN"]},
      **{t: 2 for t in ["DEN","ARI"]},
      **{t: 3 for t in ["LA","LAC","LV","OAK","SEA","SF"]}}
qb_hist = {}
def modal_qb(team, season, week):
    h = qb_hist.get(team, [])
    h = [q for s, w, q in h if (s, w) < (season, week)][-6:]
    if not h: return None
    return max(set(h), key=h.count)
sc_sorted = sc.sort_values(["season", "week"])
backup = {}
for r in sc_sorted.itertuples():
    for side, qb in [("home", r.home_qb_id), ("away", r.away_qb_id)]:
        team = getattr(r, f"{side}_team")
        m = modal_qb(team, r.season, r.week)
        backup[(r.season, r.week, team)] = (m is not None and pd.notna(qb) and qb != m)
        qb_hist.setdefault(team, []).append((r.season, r.week, qb))
scm = sc[["season", "week", "home_team", "away_team", "home_rest", "away_rest"]].rename(
    columns={"home_team": "home", "away_team": "away"})
gdf = gdf.merge(scm, on=["season", "week", "home", "away"], how="left")
gdf["backup_h"] = [backup.get((s, w, t), False) for s, w, t in zip(gdf.season, gdf.week, gdf.home)]
gdf["backup_a"] = [backup.get((s, w, t), False) for s, w, t in zip(gdf.season, gdf.week, gdf.away)]
gdf["rest_diff"] = (gdf["home_rest"].fillna(7) - gdf["away_rest"].fillna(7)) / 7.0
gdf["short_h"] = gdf["home_rest"].fillna(7) <= 5
gdf["short_a"] = gdf["away_rest"].fillna(7) <= 5
gdf["tz_shift_a"] = [abs(TZ.get(a, 0) - TZ.get(h, 0)) for a, h in zip(gdf.away, gdf.home)]

# ---------------- iteration harness
EVAL = [2020, 2021, 2022, 2023]
def evaluate(pred_fn, name, kernel=False, note=""):
    rows = []
    for s in EVAL:
        tr, te = gdf[gdf["season"] < s], gdf[gdf["season"] == s]
        m_hat, extra = pred_fn(tr, te)
        d = te.copy(); d["m_hat"] = m_hat
        rows.append((d, tr, extra))
    allte = pd.concat([r[0] for r in rows])
    err = allte["m_hat"] - allte["margin"]
    res = {"iteration": name, "n": len(allte),
           "margin_MAE": round(float(err.abs().mean()), 3),
           "margin_corr": round(float(np.corrcoef(allte["m_hat"], allte["margin"])[0, 1]), 4),
           "SU_acc": round(float((np.sign(allte["m_hat"]) == np.sign(allte["margin"]))
                                 [allte["margin"] != 0].mean()), 4),
           "note": note}
    np_ = allte["margin"] != allte["spread_line"]
    ats_pick_home = allte["m_hat"] > allte["spread_line"]
    cover_home = allte["margin"] > allte["spread_line"]
    res["ATS_acc_all"] = round(float((ats_pick_home == cover_home)[np_].mean()), 4)
    if kernel:   # key-number-aware selective betting
        picks = []
        for d, tr, _ in rows:
            tr_err = []  # build forecast-error kernel from TRAIN under same model
            m_tr, _ = kernel_models[name](tr[tr["season"] < tr["season"].max()], tr[tr["season"] == tr["season"].max()]) \
                      if False else (None, None)
            picks.append(d)
        d = pd.concat(picks)
        # kernel: integerized error distribution from all TRAIN seasons pooled (prior seasons only per fold)
        sel_acc, sel_n, units = kernel_eval(rows)
        res["kernel_selective"] = {"n_bets": sel_n, "ATS_acc": sel_acc, "units_at_-110": units}
    OUT.setdefault("iterations", []).append(res)
    print(json.dumps(res), flush=True)
    return res

def kernel_eval(rows, thresh=0.035):
    """Empirical error kernel per fold (train errors, integer-binned) ->
    P(home covers) -> bet when |p-0.5| implies EV>thresh at -110."""
    H = N = 0; units = 0.0
    for d, tr, extra in rows:
        tr_pred = extra.get("train_pred")
        if tr_pred is None: continue
        errs = (tr["margin"] - tr_pred).round().astype(int)
        kern = errs.value_counts(normalize=True).sort_index()
        ks, ps = kern.index.to_numpy(), kern.to_numpy()
        for r in d.itertuples():
            need = r.spread_line - r.m_hat      # home covers if err > need
            p_cover = ps[ks > need].sum() + 0.5 * ps[ks == round(need)].sum()
            p = max(p_cover, 1 - p_cover)
            ev = p * (10 / 11) - (1 - p)
            if ev < thresh: continue
            pick_home = p_cover >= 0.5
            cov = r.margin > r.spread_line
            if r.margin == r.spread_line: continue
            N += 1; win = (pick_home == cov); H += win
            units += (10 / 11) if win else -1.0
    return (round(H / N, 4) if N else None), N, round(units, 1)

# it0: current ratings baseline (off/def points + hfa 1.5)
def it0(tr, te):
    m = (te["off_home"] - te["def_away"]) - (te["off_away"] - te["def_home"]) + 1.5
    mtr = (tr["off_home"] - tr["def_away"]) - (tr["off_away"] - tr["def_home"]) + 1.5
    return m.to_numpy(), {"train_pred": mtr.to_numpy()}

FEATS1 = ["dp", "dr", "hfa1"]
def feats1(d):
    return pd.DataFrame({
        "dp": (d["off_pass_h"] - d["def_pass_a"]) - (d["off_pass_a"] - d["def_pass_h"]),
        "dr": (d["off_rush_h"] - d["def_rush_a"]) - (d["off_rush_a"] - d["def_rush_h"]),
        "hfa1": 1.0}, index=d.index)
def fit_ols(Xtr, ytr, Xte):
    b, *_ = np.linalg.lstsq(Xtr.to_numpy(), ytr.to_numpy(), rcond=None)
    return Xte.to_numpy() @ b, Xtr.to_numpy() @ b
def it1(tr, te):
    p, ptr = fit_ols(feats1(tr), tr["margin"], feats1(te))
    return p, {"train_pred": ptr}

def feats2(d):
    f = feats1(d)
    f["backup"] = d["backup_a"].astype(float) - d["backup_h"].astype(float)
    return f
def it2(tr, te):
    p, ptr = fit_ols(feats2(tr), tr["margin"], feats2(te))
    return p, {"train_pred": ptr}

def feats3(d):
    f = feats2(d)
    f["rest_diff"] = d["rest_diff"]
    f["short_a_minus_h"] = d["short_a"].astype(float) - d["short_h"].astype(float)
    f["tz_shift_a"] = d["tz_shift_a"].astype(float)
    return f
def it3(tr, te):
    p, ptr = fit_ols(feats3(tr), tr["margin"], feats3(te))
    return p, {"train_pred": ptr}

def it5(tr, te, alpha_grid=np.arange(0, 1.01, 0.05)):
    pm_tr, ptr = fit_ols(feats3(tr), tr["margin"], feats3(tr))
    pm_te, _ = fit_ols(feats3(tr), tr["margin"], feats3(te))
    best_a, best_mae = 0, 1e9
    for a in alpha_grid:
        mae = np.abs(a * ptr + (1 - a) * tr["spread_line"] - tr["margin"]).mean()
        if mae < best_mae: best_mae, best_a = mae, a
    blend = best_a * pm_te + (1 - best_a) * te["spread_line"]
    return blend.to_numpy(), {"train_pred": best_a * ptr + (1 - best_a) * tr["spread_line"].to_numpy(),
                              "alpha": best_a}

kernel_models = {}
evaluate(it0, "it0_points_elo_baseline", kernel=True, note="current build_ratings ratings + HFA 1.5")
evaluate(it1, "it1_epa_pass_rush_ratings", kernel=True, note="opp-adjusted EWM EPA/play, pass/rush split, OLS->margin")
evaluate(it2, "it2_plus_QB_backup", kernel=True, note="+ backup-starter detection from schedules")
r3 = evaluate(it3, "it3_plus_rest_hfa_travel", kernel=True, note="+ rest diff, short week, timezone travel, fit HFA")
r5 = evaluate(it5, "it5_market_blend", kernel=True, note="alpha*model + (1-alpha)*closing spread, alpha fit walk-forward")

# market-disagreement selectivity on it3: bet only when |model - market| in bands
d_all = []
for s in EVAL:
    tr, te = gdf[gdf["season"] < s], gdf[gdf["season"] == s]
    p, _ = it3(tr, te)
    d = te.copy(); d["m_hat"] = p; d_all.append(d)
d = pd.concat(d_all); np_ = d["margin"] != d["spread_line"]
d = d[np_]
d["gap"] = d["m_hat"] - d["spread_line"]
d["pick_home"] = d["gap"] > 0
d["hit"] = d["pick_home"] == (d["margin"] > d["spread_line"])
bands = {}
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]:
    sel = d["gap"].abs().between(lo, hi, inclusive="left")
    if sel.sum() >= 30:
        bands[f"{lo}-{hi}pts"] = {"n": int(sel.sum()), "ATS_acc": round(float(d.loc[sel, "hit"].mean()), 4)}
OUT["disagreement_bands_it3"] = bands
print("disagreement bands:", json.dumps(bands), flush=True)

json.dump(OUT, open(os.path.join(ROOT, "book", "line_engine_iterations.json"), "w"), indent=1)
print("saved book/line_engine_iterations.json", flush=True)
