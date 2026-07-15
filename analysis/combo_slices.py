#!/usr/bin/env python3
"""Resumable factor-combination search. Run repeatedly; each run does ~BUDGET s
of work, checkpoints, and exits. Phases: cache -> s1 (OLS screen) -> s2 (GBDT
confirm) -> s3 (winner ablation) -> done. State in data/analysis_cache/combo_state/."""
import json, os, sys, time, itertools
import numpy as np, pandas as pd

T0 = time.time(); BUDGET = 33.0
def timeleft(): return BUDGET - (time.time() - T0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nflvalue import ml_ranker as mlr

ST = "data/analysis_cache/combo_state"; os.makedirs(ST, exist_ok=True)
STATE_F = f"{ST}/state.json"
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"phase": "cache", "i": 0}
def save_state(): json.dump(state, open(STATE_F, "w"))

GROUPS = {
    "model_belief":   ["p_over", "z", "mean", "sd", "line", "mean_minus_line", "sd_over_line"],
    "proj_parts":     ["opp_factor", "game_script", "proj_volume", "proj_efficiency"],
    "usage_rolls":    ["roll_games", "roll_targets", "roll_target_share", "roll_carries",
                        "roll_carry_share", "roll_pass_attempts", "roll_adot", "roll_air_yards",
                        "roll_ypt", "roll_catch_rate", "roll_ypc", "roll_ypa"],
    "game_ctx":       ["team_margin", "total_line", "home", "week"],
    "situational":    ["is_birthday_week", "revenge_game"],
    "def_injuries":   ["def_out_total", "def_out_db"],
    "opp_epa":        ["opp_epa_factor"],
    "team_tendencies":["team_neutral_proe", "team_edp", "team_pace", "team_epa_play",
                        "team_shotgun_rate", "team_no_huddle_rate", "team_cpoe"],
    "ngs":            ["ngs_separation", "ngs_ay_share", "ngs_yac_aoe"],
    "redzone":        ["rz_tgt_share", "rz_carry_share"],
    "qb_oline":       ["qb_continuity", "oline_outs"],
    "player_status":  ["is_contract_year", "age_years"],
    "weather":        ["temp", "wind"],
    "chemistry":      ["shotgun_tilt_tgt", "shotgun_tilt_carry", "qb_chem_delta",
                        "key_teammate_absent", "teammate_out_boost", "opp_pressure_rate"],
    "ftn":            ["own_pa_rate", "own_motion_rate", "opp_blitz_rate", "opp_box_avg"],
    "market_pos":     [f"mkt_{m}" for m in mlr.MARKETS7] + [f"pos_{p}" for p in mlr.POSITIONS],
}
GNAMES = list(GROUPS); EVAL_SEASONS = [2022, 2023, 2024, 2025]
SPARSE = {"ngs", "chemistry", "ftn", "weather"}

def subsets_list():
    rng = np.random.default_rng(20260714)
    subs = [(g,) for g in GNAMES]
    subs += list(itertools.combinations(GNAMES, 2))
    subs += [tuple(x for x in GNAMES if x != g) for g in GNAMES]
    subs += [tuple(GNAMES)]
    seen = set(subs); target = len(subs) + 1500
    while len(subs) < target:
        k = int(rng.integers(3, 10))
        s = tuple(sorted(str(x) for x in rng.choice(GNAMES, size=k, replace=False)))
        if s not in seen: seen.add(s); subs.append(s)
    return subs

def col_layout():
    idx, cur, names = {}, 0, []
    for g in GNAMES:
        for c in GROUPS[g]: idx[(g, c)] = cur; cur += 1; names.append(c)
        if g in SPARSE: idx[(g, f"_{g}_miss")] = cur; cur += 1; names.append(f"_{g}_miss")
    idx[("_const", "_const")] = cur
    return idx, cur + 1
COLIDX, NCOLS = col_layout()
def cols_for(gset):
    ii = []
    for g in gset:
        ii += [COLIDX[(g, c)] for c in GROUPS[g]]
        if g in SPARSE: ii.append(COLIDX[(g, f"_{g}_miss")])
    ii.append(COLIDX[("_const", "_const")])
    return np.array(sorted(set(ii)))

def grade(test, p):
    leans = mlr.rank_and_grade(test, p)
    top1 = leans.groupby(["season", "week", "game_id"]).head(1)
    return len(leans), int(leans["ml_hit"].sum()), len(top1), int(top1["ml_hit"].sum())

# ---------------------------------------------------------------- phases
if state["phase"] == "cache":
    frame = pd.read_parquet(f"{ROOT}/data/ml_frame.parquet")
    allcols = [c for g in GNAMES for c in GROUPS[g]]
    for s in EVAL_SEASONS:
        tr, te = frame[frame["season"] < s], frame[frame["season"] == s]
        med = tr[allcols].median(numeric_only=True).fillna(0.0)
        def mat(df):
            X = np.empty((len(df), NCOLS), dtype=np.float32)
            for g in GNAMES:
                sub = df[GROUPS[g]].astype(float)
                if g in SPARSE:
                    X[:, COLIDX[(g, f"_{g}_miss")]] = sub.isna().all(axis=1).astype(float)
                for c in GROUPS[g]:
                    X[:, COLIDX[(g, c)]] = sub[c].fillna(med[c])
            X[:, -1] = 1.0
            return X
        Xtr, Xte = mat(tr), mat(te)
        mu, sd = Xtr[:, :-1].mean(0), Xtr[:, :-1].std(0) + 1e-9
        Xtr[:, :-1] = (Xtr[:, :-1] - mu) / sd; Xte[:, :-1] = (Xte[:, :-1] - mu) / sd
        np.save(f"{ST}/Xtr_{s}.npy", Xtr); np.save(f"{ST}/Xte_{s}.npy", Xte)
        np.save(f"{ST}/ytr_{s}.npy", tr["y_over"].to_numpy(np.float32))
        te[["season", "week", "game_id", "player_id", "market", "y_over"]].reset_index(
            drop=True).to_parquet(f"{ST}/te_{s}.parquet")
    state.update(phase="s1", i=0); save_state(); print("cache done", flush=True)

if state["phase"] == "s1":
    subs = subsets_list()
    cache = {s: (np.load(f"{ST}/Xtr_{s}.npy"), np.load(f"{ST}/ytr_{s}.npy"),
                 np.load(f"{ST}/Xte_{s}.npy"), pd.read_parquet(f"{ST}/te_{s}.parquet"))
             for s in EVAL_SEASONS}
    outf = open(f"{ST}/s1.jsonl", "a")
    i = state["i"]
    while i < len(subs) and timeleft() > 3:
        cc = cols_for(subs[i]); H = N = H1 = N1 = 0
        for s in EVAL_SEASONS:
            Xtr, ytr, Xte, te = cache[s]
            beta, *_ = np.linalg.lstsq(Xtr[:, cc], ytr, rcond=None)
            n, h, n1, h1 = grade(te, np.clip(Xte[:, cc] @ beta, 0.01, 0.99))
            H += h; N += n; H1 += h1; N1 += n1
        outf.write(json.dumps({"groups": list(subs[i]), "k": len(subs[i]),
                               "hit": H / N, "top1": H1 / N1, "n": N}) + "\n")
        i += 1
    outf.close()
    state["i"] = i
    if i >= len(subs): state.update(phase="s2", i=0)
    save_state(); print(f"s1 progress {i}/{len(subs)}", flush=True)

elif state["phase"] in ("s2", "s3a", "s3b"):
    S1 = pd.read_json(f"{ST}/s1.jsonl", lines=True)
    S1["groups"] = S1["groups"].map(tuple)
    top_sets = list(dict.fromkeys(S1.sort_values("hit", ascending=False)["groups"].head(10)))
    frame = pd.read_parquet(f"{ROOT}/data/ml_frame.parquet")
    donef = f"{ST}/gbdt.jsonl"
    done = set()
    if os.path.exists(donef):
        done = {(tuple(json.loads(l)["groups"]), json.loads(l)["season"]) for l in open(donef)}

    def gbdt_fit_one(gset, season):
        cols = [c for g in gset for c in GROUPS[g] if not c.startswith(("mkt_", "pos_"))]
        if "market_pos" in gset:
            cols += [f"mkt_{m}" for m in mlr.MARKETS7] + [f"pos_{p}" for p in mlr.POSITIONS]
        import nflvalue.ml_ranker as _m
        old_nf, old_fc = _m.NUMERIC_FEATURES, _m.feature_columns
        try:
            _m.feature_columns = lambda: cols
            tr = frame[frame["season"] < season]
            te = frame[frame["season"] == season].reset_index(drop=True)
            model = _m.MLRanker(model="gbdt").fit(tr, tr["y_over"])
            p = model.predict_p_over(te)
            n, h, n1, h1 = grade(te, p)
            return {"groups": list(gset), "season": season, "n": n, "h": h, "n1": n1, "h1": h1}
        finally:
            _m.NUMERIC_FEATURES, _m.feature_columns = old_nf, old_fc

    if state["phase"] == "s2":
        configs = list(dict.fromkeys(
            top_sets + [tuple(GNAMES), ("model_belief",), ("model_belief", "market_pos")]))
    else:
        g_agg = {}
        for l in open(donef):
            r = json.loads(l); k = tuple(r["groups"])
            a = g_agg.setdefault(k, [0, 0]); a[0] += r["h"]; a[1] += r["n"]
        s2_keys = [k for k in g_agg]
        winner = max(s2_keys, key=lambda k: g_agg[k][0] / g_agg[k][1])
        json.dump(list(winner), open(f"{ST}/winner.json", "w"))
        if state["phase"] == "s3a":
            configs = [tuple(x for x in winner if x != g) for g in winner]
        else:
            configs = [tuple(sorted(winner + (g,))) for g in GNAMES if g not in winner]

    todo = [(c, s) for c in configs for s in EVAL_SEASONS if (c, s) not in done]
    outf = open(donef, "a")
    ran = 0
    for c, s in todo:
        if timeleft() < 8: break
        outf.write(json.dumps(gbdt_fit_one(c, s)) + "\n"); outf.flush(); ran += 1
    outf.close()
    if ran == len(todo):
        nxt = {"s2": "s3a", "s3a": "s3b", "s3b": "done"}[state["phase"]]
        state.update(phase=nxt, i=0)
    save_state(); print(f"{state['phase']} ran {ran}, remaining {len(todo) - ran}", flush=True)

if state["phase"] == "done":
    print("PHASES COMPLETE", flush=True)
