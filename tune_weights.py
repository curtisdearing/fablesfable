#!/usr/bin/env python3
"""Walk-forward composite-weight tuning over every season on disk (2019->now).

THE HONESTY CONTRACT (PREMORTEM.md: selection bias + multiple comparisons):
  * "Best weights on all the data, scored on the same data" is self-deception.
    This script SELECTS each eval season's config using ONLY prior seasons,
    then scores it out-of-sample on that season. The number reported per
    season is therefore an estimate of live performance of the *procedure*,
    not a curve-fit trophy.
  * The shipped 2026 config = argmax over ALL completed seasons pooled --
    chosen once, recorded in docs/decisions_p3-5.md, applied via config.json.
  * "Profitable" is proxied by directional hit rate at the tool's SYNTHETIC
    trailing-mean lines vs the 52.38% breakeven a real -110 price would
    demand. No historical prop prices exist free; live CLV remains the real
    referendum (Phase 3).

What is tuned (the ranking layer only -- projections stay untouched):
    conf_share        confidence-vs-matchup split (edge stays fixed at 0.5
                      and dominant whenever real prices exist; with no
                      market the two remaining terms renormalize, so this
                      ratio is what actually ordered historical leans)
    z_cap             |z| that earns full confidence credit
    low_conf_mult     composite penalty for low-confidence markets
    markets           all seven vs the four honestly-modelable core markets
                      (spec §2 ordered them for a reason)

Stage 1 (slow, cached): one weight-INDEPENDENT component frame -- every
candidate of every week with {z, model_prob, opp_factor, game_script,
low_confidence, side, hit}. Weights only reorder these, so Stage 2 evaluates
a config by vectorized re-scoring + re-ranking in milliseconds. A consistency
check asserts the vectorized scorer matches composite.score_candidate exactly
on a sample before any search runs.

Run:
    python3 tune_weights.py --stage components --seasons 2019 2020 ...
    python3 tune_weights.py --stage search
Writes: data/weight_tuning.json + reports/weight_tuning.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from nflvalue import config as cfgmod
from nflvalue.candidates import WeekInputs, enumerate_candidates
from nflvalue.composite import DEFAULT_PARAMS, YES_ONLY_MARKETS, score_candidate
from nflvalue.projection import MARKETS

FRAME_PATH = os.path.join(cfgmod.DATA_DIR, "tuning_components.parquet")
BREAKEVEN = 0.5238
CORE4 = ("receiving_yards", "receptions", "rushing_yards", "passing_yards")

GRID = {
    "conf_share": [0.4, 0.5, 0.6, 0.7, 0.8],
    "z_cap": [1.5, 2.0, 2.5],
    "low_conf_mult": [0.6, 0.8, 1.0],
    "markets": ["all", "core4"],
}
MIN_TRAIN_LEANS = 1000
TOP_N, MAX_PER_PLAYER = 5, 2


# --------------------------------------------------------------------------- #
# Stage 1: the weight-independent component frame
# --------------------------------------------------------------------------- #
def build_components(inputs: WeekInputs, seasons: List[int],
                     out_path: str = FRAME_PATH, append: bool = False) -> pd.DataFrame:
    import lean_backtest as lb
    from nflvalue.candidates import synthetic_lines

    sd_map = lb.precompute_sds(inputs, list(MARKETS))
    synth_map = {m: synthetic_lines(inputs, m) for m in MARKETS}
    min_usage = (cfgmod.load_config().get("candidates") or {}).get("min_usage")

    rows = []
    for season in seasons:
        weeks = sorted(inputs.schedules[
            (inputs.schedules["season"] == season)
            & (inputs.schedules["game_type"] == "REG")]["week"].unique().tolist())
        for wk in weeks:
            try:
                cands = enumerate_candidates(season, wk, inputs=inputs,
                                             min_usage=min_usage,
                                             sd_by_market=sd_map.get((season, wk), {}),
                                             synth_by_market=synth_map)
            except ValueError:
                continue
            if cands.empty:
                continue
            actuals = lb._actuals_for_week(inputs.pw, season, wk)
            for c in cands.to_dict("records"):
                p_over = c.get("p_over")
                if p_over is None:
                    continue
                side = "over" if (c["market"] in YES_ONLY_MARKETS or p_over >= 0.5) else "under"
                graded = lb.grade({**c, "side": side}, actuals)
                if graded is None:
                    continue
                comps = c.get("components") or {}
                rows.append({
                    "season": season, "week": wk, "game_id": c["game_id"],
                    "player_id": c["player_id"], "market": c["market"], "side": side,
                    "z": (c["mean"] - c["line"]) / max(c["sd"], 1e-6),
                    "model_prob": p_over if side == "over" else 1 - p_over,
                    "opp_factor": float(comps.get("opp_factor", 1.0) or 1.0),
                    "game_script": float(comps.get("game_script", 1.0) or 1.0),
                    "low_confidence": bool(c.get("low_confidence")),
                    "hit": int(graded["hit"]),
                })
    frame = pd.DataFrame(rows)
    if append and os.path.exists(out_path):
        old = pd.read_parquet(out_path)
        frame = (pd.concat([old, frame], ignore_index=True)
                 .drop_duplicates(subset=["season", "week", "game_id", "player_id", "market"]))
    frame.to_parquet(out_path, index=False)
    return frame


# --------------------------------------------------------------------------- #
# Stage 2: vectorized scoring + ranking (must match composite.score_candidate)
# --------------------------------------------------------------------------- #
def score_frame(f: pd.DataFrame, conf_share: float, z_cap: float,
                low_conf_mult: float) -> np.ndarray:
    d = np.where(f["side"].to_numpy() == "over", 1.0, -1.0)
    conf = np.clip(np.minimum(np.abs(f["z"].to_numpy()), z_cap) / z_cap, 0, 1)
    conf = np.where(f["model_prob"].to_numpy() < 0.5, 0.0, conf)
    opp_sub = np.clip(0.5 + d * (f["opp_factor"].to_numpy() - 1.0)
                      / DEFAULT_PARAMS["opp_factor_span"] * 0.5, 0, 1)
    script_sub = np.clip(0.5 + d * (f["game_script"].to_numpy() - 1.0)
                         / DEFAULT_PARAMS["script_span"] * 0.5, 0, 1)
    matchup = (opp_sub + script_sub + 0.5) / 3.0
    wc, wm = conf_share, 1.0 - conf_share
    comp = 100.0 * (wc * conf + wm * matchup)
    return np.where(f["low_confidence"].to_numpy(), comp * low_conf_mult, comp)


def _assert_scorer_matches_composite(frame: pd.DataFrame) -> None:
    sample = frame.sample(min(200, len(frame)), random_state=7)
    for conf_share, z_cap, lcm in ((0.6, 2.0, 0.8), (0.4, 1.5, 1.0)):
        vec = score_frame(sample, conf_share, z_cap, lcm)
        for v, row in zip(vec, sample.to_dict("records")):
            cand = {"market": row["market"],
                    "mean": row["z"], "sd": 1.0, "line": 0.0,   # reconstructs the same z
                    "p_over": row["model_prob"] if row["side"] == "over" else 1 - row["model_prob"],
                    "p_under": 1 - (row["model_prob"] if row["side"] == "over" else 1 - row["model_prob"]),
                    "components": {"opp_factor": row["opp_factor"],
                                   "game_script": row["game_script"]},
                    "prices": None, "low_confidence": row["low_confidence"]}
            ref = score_candidate(cand, weights={"edge": 0.5,
                                                 "confidence": 0.5 * conf_share,
                                                 "matchup": 0.5 * (1 - conf_share)},
                                  params={"z_cap": z_cap, "low_confidence_mult": lcm})
            if abs(ref["composite"] - v) > 0.51:   # score_candidate rounds to 2dp
                raise AssertionError(
                    f"vectorized scorer drifted from composite.score_candidate: "
                    f"{v:.2f} vs {ref['composite']:.2f} for {row}")


class _FastEval:
    """One presorted numpy sweep per config, tallying (hits, picks) PER SEASON.

    Any train pool is then just a sum over prior seasons' tallies, so the
    whole walk-forward grid costs ~one sweep per config instead of one per
    (config x pool). Selection semantics are identical to shortlist.rank_game:
    composite desc, deterministic (player_id, market) tie-break, per-player
    cap, top 5 per game."""

    def __init__(self, frame: pd.DataFrame):
        self.f = frame.reset_index(drop=True)
        gkey, _ = pd.factorize(list(zip(self.f["season"], self.f["week"], self.f["game_id"])))
        self.gkey = gkey.astype(np.int64)
        self.player, _ = pd.factorize(self.f["player_id"])
        self.market_rank, _ = pd.factorize(self.f["market"].astype(str))
        self.season = self.f["season"].to_numpy()
        self.hit = self.f["hit"].to_numpy()
        self.core4_mask = self.f["market"].isin(CORE4).to_numpy()

    def per_season(self, cfg: Dict) -> Dict[int, tuple]:
        comp = score_frame(self.f, cfg["conf_share"], cfg["z_cap"], cfg["low_conf_mult"])
        mask = np.ones(len(self.f), bool) if cfg["markets"] == "all" else self.core4_mask
        idx = np.flatnonzero(mask)
        order = idx[np.lexsort((self.market_rank[idx], self.player[idx],
                                -comp[idx], self.gkey[idx]))]
        tally: Dict[int, list] = {}
        cur_g, taken, per_player = -1, 0, {}
        for i in order:
            g = self.gkey[i]
            if g != cur_g:
                cur_g, taken, per_player = g, 0, {}
            if taken >= TOP_N:
                continue
            p = self.player[i]
            if per_player.get(p, 0) >= MAX_PER_PLAYER:
                continue
            per_player[p] = per_player.get(p, 0) + 1
            taken += 1
            t = tally.setdefault(int(self.season[i]), [0, 0])
            t[0] += int(self.hit[i])
            t[1] += 1
        return {s: (h, n) for s, (h, n) in tally.items()}


def _pool(per_season: Dict[int, tuple], seasons: List[int]) -> Dict:
    h = sum(per_season.get(s, (0, 0))[0] for s in seasons)
    n = sum(per_season.get(s, (0, 0))[1] for s in seasons)
    return {"n": int(n), "hit_rate": round(h / n, 4) if n else None}


def eval_config(frame: pd.DataFrame, cfg: Dict) -> Dict:
    ps = _FastEval(frame).per_season(cfg)
    return _pool(ps, sorted(ps))


def search(frame: pd.DataFrame) -> Dict:
    _assert_scorer_matches_composite(frame)
    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    seasons = sorted(int(s) for s in frame["season"].unique())
    fe = _FastEval(frame)
    tallies = [fe.per_season(cfg) for cfg in configs]
    default_cfg = {"conf_share": 0.6, "z_cap": 2.0, "low_conf_mult": 0.8, "markets": "all"}
    default_tally = fe.per_season(default_cfg)

    walk_forward = []
    for eval_season in seasons[1:]:
        train_seasons = [s for s in seasons if s < eval_season]
        scored = []
        for cfg, tally in zip(configs, tallies):
            r = _pool(tally, train_seasons)
            if r["n"] >= MIN_TRAIN_LEANS and r["hit_rate"] is not None:
                scored.append((r["hit_rate"], r["n"], cfg, tally))
        if not scored:
            continue
        scored.sort(key=lambda x: (-x[0], -x[1], json.dumps(x[2], sort_keys=True)))
        rate, _, best_cfg, best_tally = scored[0]
        walk_forward.append({"eval_season": eval_season,
                             "chosen_on_train": best_cfg,
                             "train_hit_rate": rate,
                             "oos": _pool(best_tally, [eval_season]),
                             "default_config_oos": _pool(default_tally, [eval_season])})

    pooled = []
    for cfg, tally in zip(configs, tallies):
        r = _pool(tally, seasons)
        if r["hit_rate"] is not None:
            pooled.append({**cfg, **r})
    pooled.sort(key=lambda x: (-x["hit_rate"], -x["n"]))

    return {"walk_forward": walk_forward, "pooled_top10": pooled[:10],
            "ship_for_2026": pooled[0],
            "grid_size": len(configs),
            "note": ("walk_forward rows are OUT-OF-SAMPLE (config chosen on prior "
                     "seasons only); pooled_top10 is in-sample across 2019-2025 and "
                     "shown for transparency, with ship_for_2026 = pooled argmax. "
                     "Directional hit rate at synthetic lines; breakeven proxy "
                     f"{BREAKEVEN:.2%}.")}


def render_md(res: Dict, frame: pd.DataFrame) -> str:
    lines = [
        "# Composite-weight tuning — walk-forward, 2019–2025",
        "",
        "**Leans, not locks.** " + res["note"] + " 1-800-GAMBLER.",
        "",
        "## Out-of-sample: each season scored with weights chosen ONLY from prior seasons",
        "",
        "| Season | chosen on train | train hit | OOS hit (n) | old-default OOS |",
        "|---|---|---|---|---|",
    ]
    for w in res["walk_forward"]:
        c = w["chosen_on_train"]
        lines.append(
            f"| {w['eval_season']} | conf {c['conf_share']}, z_cap {c['z_cap']}, "
            f"lcm {c['low_conf_mult']}, {c['markets']} | {w['train_hit_rate']:.1%} "
            f"| **{w['oos']['hit_rate']:.1%}** ({w['oos']['n']}) "
            f"| {w['default_config_oos']['hit_rate']:.1%} |")
    lines += ["", "## Pooled 2019–2025 (in-sample, for transparency)", "",
              "| conf_share | z_cap | low_conf_mult | markets | n | hit |",
              "|---|---|---|---|---|---|"]
    for p in res["pooled_top10"]:
        lines.append(f"| {p['conf_share']} | {p['z_cap']} | {p['low_conf_mult']} "
                     f"| {p['markets']} | {p['n']} | {p['hit_rate']:.1%} |")
    s = res["ship_for_2026"]
    lines += ["", f"**Shipped for 2026:** conf_share {s['conf_share']}, z_cap {s['z_cap']}, "
              f"low_conf_mult {s['low_conf_mult']}, markets={s['markets']} "
              f"(pooled {s['hit_rate']:.1%} over n={s['n']}). Live weights stay "
              "edge-dominant (0.5) whenever a real price exists; this tunes the "
              "other half and the no-market ordering.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["components", "search"], required=True)
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    ap.add_argument("--pw"), ap.add_argument("--opd"), ap.add_argument("--tw"), ap.add_argument("--sched")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    if args.stage == "components":
        if args.pw:
            inputs = WeekInputs(pd.read_parquet(args.pw), pd.read_parquet(args.opd),
                                pd.read_parquet(args.tw), pd.read_parquet(args.sched))
        else:
            from nflvalue.candidates import build_week_inputs
            inputs = build_week_inputs()
        seasons = args.seasons or sorted(inputs.pw["season"].unique().tolist())
        frame = build_components(inputs, seasons, append=args.append)
        print(f"components: {len(frame):,} rows, seasons {sorted(frame['season'].unique())}")
        return

    frame = pd.read_parquet(FRAME_PATH)
    res = search(frame)
    cfgmod.save_json(os.path.join(cfgmod.DATA_DIR, "weight_tuning.json"), res)
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "weight_tuning.md"), "w") as f:
        f.write(render_md(res, frame))
    print(json.dumps({"ship_for_2026": res["ship_for_2026"],
                      "oos_by_season": [(w["eval_season"], w["oos"]) for w in res["walk_forward"]]},
                     indent=1))


if __name__ == "__main__":
    main()
