#!/usr/bin/env python3
"""Monte Carlo search for a retrospective 'perfect' (100%) projection.

Part A — GAME layer: 1,342 real games 2019-2023 (data/backtest_games.json).
  Random-sample factor-weight vectors; pick ATS side per game; measure:
  full-slate accuracy, top-N confidence accuracy, and the deepest
  100%-correct confidence-ranked run per config. Honesty check: refit
  in-sample 2019-2022, evaluate 2023 out-of-sample.

Part B — PROP layer: 1,360 real 2025 leans (data/lean_replay_2025.json).
  Exhaustive scan over selection rules (market subset x side x composite
  threshold x rank cutoff); find rules that graded 100% and bootstrap
  their forward-looking hit-rate distribution (the actual Monte Carlo).

Run from the fablesfable repo root.
"""
import json, itertools, sys
import numpy as np

rng = np.random.default_rng(42)
OUT = {}

# --------------------------------------------------------------- Part A data
DIV = {
    "BUF":"AFCE","MIA":"AFCE","NE":"AFCE","NYJ":"AFCE",
    "BAL":"AFCN","CIN":"AFCN","CLE":"AFCN","PIT":"AFCN",
    "HOU":"AFCS","IND":"AFCS","JAX":"AFCS","TEN":"AFCS",
    "DEN":"AFCW","KC":"AFCW","LV":"AFCW","OAK":"AFCW","LAC":"AFCW",
    "DAL":"NFCE","NYG":"NFCE","PHI":"NFCE","WAS":"NFCE",
    "CHI":"NFCN","DET":"NFCN","GB":"NFCN","MIN":"NFCN",
    "ATL":"NFCS","CAR":"NFCS","NO":"NFCS","TB":"NFCS",
    "ARI":"NFCW","LA":"NFCW","SEA":"NFCW","SF":"NFCW",
}

games = [g for g in json.load(open("data/backtest_games.json")) if g.get("ready")]
games.sort(key=lambda g: (g["season"], g["week"], g["gameday"]))

# rest days + revenge flags, walking the schedule chronologically
last_played, met = {}, {}
rows = []
import datetime as dt
for g in games:
    d = dt.date.fromisoformat(g["gameday"])
    h, a, season = g["home"], g["away"], g["season"]
    rest_h = (d - last_played[(season,h)]).days if (season,h) in last_played else 7
    rest_a = (d - last_played[(season,a)]).days if (season,a) in last_played else 7
    last_played[(season,h)] = d; last_played[(season,a)] = d
    key = (season, tuple(sorted((h,a))))
    rev_h = rev_a = 0.0
    if key in met:                       # rematch: loser of 1st meeting = revenge
        prev_winner = met[key]
        if prev_winner == a: rev_h = 1.0
        elif prev_winner == h: rev_a = 1.0
    margin = g["home_score"] - g["away_score"]
    met[key] = h if margin > 0 else (a if margin < 0 else None)

    matchup = (g["off_home"] - g["def_away"]) - (g["off_away"] - g["def_home"])
    cover = margin + g["spread_line"]     # nflverse: spread_line = home handicap? check sign below
    rows.append(dict(season=season, week=g["week"],
        matchup=matchup, home=1.0, rest=(rest_h-rest_a)/7.0,
        revenge=rev_h-rev_a, div=1.0 if DIV[h]==DIV[a] else 0.0,
        spread=g["spread_line"], total=g["total_line"], margin=margin,
        ml_home=g["home_ml"]))

# Resolve spread sign convention: pick the sign that makes favorites cover ~50%
m = np.array([r["margin"] for r in rows]); s = np.array([r["spread"] for r in rows])
convA = np.mean((m + s) > 0)   # cover if margin + line > 0
convB = np.mean((m - s) > 0)
# spread_line in nflverse = expected home margin (home -3 favorite => +3.0)
# home covers when margin > spread_line  => use (m - s)
cover_home = (m - s) > 0
push = (m - s) == 0
OUT["spread_sign_check"] = {"P(margin+s>0)": round(float(convA),4), "P(margin-s>0)": round(float(convB),4)}

FCOLS = ["matchup","home","rest","revenge","div","mkt_spread"]
X = np.array([[r["matchup"], r["home"], r["rest"], r["revenge"], r["div"], -r["spread"]] for r in rows])
X = ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)   # z-score each factor
y = np.where(cover_home, 1, -1).astype(np.int8)   # +1 home covered
keep = ~push
Xk, yk = X[keep], y[keep]
seasons = np.array([r["season"] for r in rows])[keep]
N = len(yk)
OUT["game_n"] = int(N)

def evaluate(W, Xe, ye):
    """W: (k,6) weights. Returns per-config: full acc, deepest 100% run, top-N accs."""
    S = Xe @ W.T                                   # (n, k) scores
    pred = np.where(S >= 0, 1, -1)
    correct = (pred == ye[:, None])                # (n, k)
    full_acc = correct.mean(0)
    conf = np.abs(S)
    order = np.argsort(-conf, axis=0)              # rank games by confidence per config
    corr_sorted = np.take_along_axis(correct, order, axis=0)
    # deepest all-correct prefix
    first_wrong = np.argmax(~corr_sorted, axis=0)
    all_right = corr_sorted.all(0)
    run100 = np.where(all_right, corr_sorted.shape[0], first_wrong)
    top = {n: corr_sorted[:n].mean(0) for n in (10, 25, 50, 100, 250)}
    return full_acc, run100, top

BATCH, TOTAL = 4000, 200000
best = dict(full=-1, run=-1)
acc_all, run_all = [], []
best_W_full = best_W_run = None
for start in range(0, TOTAL, BATCH):
    W = rng.normal(size=(BATCH, 6)).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    fa, r1, top = evaluate(W, Xk, yk)
    acc_all.append(fa); run_all.append(r1)
    i = int(np.argmax(fa)); j = int(np.argmax(r1))
    if fa[i] > best["full"]:
        best["full"], best["full_top"] = float(fa[i]), {n: float(top[n][i]) for n in top}
        best_W_full = W[i].copy()
    if r1[j] > best["run"]:
        best["run"] = int(r1[j]); best_W_run = W[j].copy()
        best["run_full_acc"] = float(fa[j])
acc_all = np.concatenate(acc_all); run_all = np.concatenate(run_all)

OUT["game_mc"] = {
    "configs_sampled": TOTAL,
    "full_slate_acc": {"max": round(float(acc_all.max()),4),
                       "p99": round(float(np.percentile(acc_all,99)),4),
                       "median": round(float(np.median(acc_all)),4)},
    "best_full_weights": dict(zip(FCOLS, np.round(best_W_full,3).tolist())),
    "best_full_topN": {k: round(v,4) for k,v in best["full_top"].items()},
    "deepest_100pct_run": {"max_games": int(run_all.max()),
                           "median": int(np.median(run_all)),
                           "p99": int(np.percentile(run_all,99)),
                           "weights": dict(zip(FCOLS, np.round(best_W_run,3).tolist())),
                           "that_config_full_acc": round(best["run_full_acc"],4)},
    "pct_configs_beating_5238": round(float((acc_all > 0.5238).mean()),4),
}

# ---- honesty check: pick best config on 2019-2022, grade on 2023
tr, te = seasons < 2023, seasons == 2023
best_tr, Wh_best = -1.0, None
rng3 = np.random.default_rng(7)
for _ in range(25):
    Wh = rng3.normal(size=(BATCH, 6)).astype(np.float32)
    Wh /= np.linalg.norm(Wh, axis=1, keepdims=True)
    fa_tr, _, _ = evaluate(Wh, Xk[tr], yk[tr])
    i = int(np.argmax(fa_tr))
    if fa_tr[i] > best_tr:
        best_tr, Wh_best = float(fa_tr[i]), Wh[i].copy()
Wh = Wh_best[None, :]; ibest = 0; fa_tr = np.array([best_tr])
fa_te, run_te, top_te = evaluate(Wh[ibest:ibest+1], Xk[te], yk[te])
OUT["game_oos"] = {
    "train_acc_2019_22": round(float(fa_tr[ibest]),4),
    "test_acc_2023": round(float(fa_te[0]),4),
    "test_n": int(te.sum()),
    "weights": dict(zip(FCOLS, np.round(Wh[ibest],3).tolist())),
}

# ---- factor attribution across elite configs (top 0.1% by full acc)
thr = np.percentile(acc_all, 99.9)
OUT["game_factor_pull"] = {}
# re-sample and collect elite weights (memory-light second pass)
elite = []
rng2 = np.random.default_rng(42)
for start in range(0, TOTAL, BATCH):
    W = rng2.normal(size=(BATCH, 6)).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    fa, _, _ = evaluate(W, Xk, yk)
    elite.append(W[fa >= thr])
elite = np.vstack(elite)
OUT["game_factor_pull"] = {c: {"mean_w": round(float(elite[:,i].mean()),3),
                               "share_|w|": round(float(np.abs(elite[:,i]).mean()/np.abs(elite).mean(0).sum()),3)}
                           for i,c in enumerate(FCOLS)}
OUT["game_elite_n"] = int(len(elite))

# univariate: each factor alone vs cover
OUT["game_univariate_acc"] = {c: round(float(max((np.sign(Xk[:,i])==yk).mean(),
                                                  (np.sign(-Xk[:,i])==yk).mean())),4)
                              for i,c in enumerate(FCOLS)}

# --------------------------------------------------------------- Part B props
lr = json.load(open("data/lean_replay_2025.json"))
lean = lr["lean_rows"]
mkts = sorted(set(r["market"] for r in lean))
M = np.array([mkts.index(r["market"]) for r in lean])
SIDE = np.array([1 if r["side"]=="over" else 0 for r in lean])
COMP = np.array([r["composite"] for r in lean], dtype=float)
RANK = np.array([r["rank"] for r in lean])
WEEK = np.array([r["week"] for r in lean])
HIT = np.array([bool(r["hit"]) for r in lean])
OUT["prop_n"] = int(len(lean)); OUT["prop_markets"] = mkts

results = []
comp_grid = np.percentile(COMP, [0,50,75,90,95,98])
for msub in range(1, 2**len(mkts)):
    msel = np.isin(M, [i for i in range(len(mkts)) if msub>>i & 1])
    for side in (0,1,2):
        ssel = msel if side==2 else (msel & (SIDE==side))
        for c in comp_grid:
            for rk in (1,2,3,5,99):
                sel = ssel & (COMP>=c) & (RANK<=rk)
                n = int(sel.sum())
                if n < 8: continue
                hr = float(HIT[sel].mean())
                results.append((hr, n, msub, side, float(c), rk))
results.sort(key=lambda t: (-t[0], -t[1]))
perfect = [r for r in results if r[0] == 1.0]
perfect.sort(key=lambda t: -t[1])

def rule_desc(msub, side, c, rk):
    mm = [mkts[i] for i in range(len(mkts)) if msub>>i & 1]
    sd = {0:"under only",1:"over only",2:"both sides"}[side]
    return {"markets": mm, "side": sd, "composite_min": round(c,1), "rank_max": int(rk)}

OUT["prop_scan"] = {
    "rules_scanned": len(results),
    "best_overall": [{"hit_rate": round(r[0],4), "n": r[1], **rule_desc(*r[2:])} for r in results[:5]],
    "perfect_rules_found": len(perfect),
    "largest_perfect": [{"n": r[1], **rule_desc(*r[2:])} for r in perfect[:5]],
}

# ---- Monte Carlo the biggest perfect rule + the best large-n rule:
# bootstrap weeks (resample 18 weeks with replacement) -> distribution of hit rate
def week_bootstrap(sel, iters=20000):
    hrs = np.empty(iters)
    weeks = np.unique(WEEK)
    idx_by_week = {w: np.where(sel & (WEEK==w))[0] for w in weeks}
    for it in range(iters):
        pick = rng.choice(weeks, size=len(weeks), replace=True)
        idx = np.concatenate([idx_by_week[w] for w in pick])
        hrs[it] = HIT[idx].mean() if len(idx) else np.nan
    return hrs[~np.isnan(hrs)]

mc_out = {}
if perfect:
    hr, n, msub, side, c, rk = perfect[0]
    sel = np.isin(M, [i for i in range(len(mkts)) if msub>>i & 1])
    if side!=2: sel &= (SIDE==side)
    sel &= (COMP>=c) & (RANK<=rk)
    hrs = week_bootstrap(sel)
    # posterior predictive: Beta(1+w,1+l) then binomial next-season n picks
    w_ = int(HIT[sel].sum()); l_ = int(sel.sum())-w_
    post = rng.beta(1+w_, 1+l_, 20000)
    mc_out["biggest_100pct_rule"] = {
        **rule_desc(msub, side, c, rk), "graded": f"{w_}-{l_}",
        "bootstrap_hit_rate": {"mean": round(float(hrs.mean()),4),
                               "p5": round(float(np.percentile(hrs,5)),4)},
        "bayes_true_rate": {"mean": round(float(post.mean()),4),
                            "p5": round(float(np.percentile(post,5)),4)},
        "P(next_20_all_hit)": round(float(np.mean(post**20)),4),
        "P(beats_breakeven_5238_longrun)": round(float((post>0.5238).mean()),4),
    }
# best rule with n>=100 (volume + quality)
big = [r for r in results if r[1] >= 100][:1]
if big:
    hr, n, msub, side, c, rk = big[0]
    sel = np.isin(M, [i for i in range(len(mkts)) if msub>>i & 1])
    if side!=2: sel &= (SIDE==side)
    sel &= (COMP>=c) & (RANK<=rk)
    hrs = week_bootstrap(sel)
    w_ = int(HIT[sel].sum()); l_ = int(sel.sum())-w_
    post = rng.beta(1+w_, 1+l_, 20000)
    mc_out["best_volume_rule_n>=100"] = {
        **rule_desc(msub, side, c, rk), "graded": f"{w_}-{l_}",
        "hit_rate": round(hr,4),
        "bootstrap": {"mean": round(float(hrs.mean()),4), "p5": round(float(np.percentile(hrs,5)),4)},
        "P(beats_breakeven_5238_longrun)": round(float((post>0.5238).mean()),4),
    }
OUT["prop_mc"] = mc_out

json.dump(OUT, open("book/mc_game_layer.json","w"), indent=1)
print(json.dumps(OUT, indent=1))
