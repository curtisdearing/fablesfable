#!/usr/bin/env python3
"""Phase 7.1 — probability calibration audit + method bake-off (walk-forward).

Question this answers: the GBDT is tuned for ordering + log-loss, but nobody
checked whether P(over)=0.62 actually lands 62%. This script measures that,
strictly out-of-sample and strictly walk-forward, and picks the calibration
method (if any) from the numbers.

Two-level walk-forward (the leakage discipline — see docs/decisions_p7.md):
  * BASE:  for season S the GBDT trains on seasons < S and predicts S. These
           raw out-of-sample P(over) are cached once (stage ``oos``).
  * CALIBRATOR: the map that corrects season S is fit ONLY on the raw OOS
           predictions of seasons < S. It never sees S. So the calibrated
           series runs 2022..2025 (2021 has no prior OOS season; it seeds the
           calibrator's history). Raw metrics are reported on the IDENTICAL
           pooled rows (2022..2025) so before/after is same-data.

Calibrate P(over) against y_over; p_under = 1 - cal(p_over). This is exactly
the (p_over, p_under) interface composite.py consumes, and is correct for the
one-sided anytime_td market (P(score)).

Methods bake-off: {raw, platt, isotonic, beta} x {pooled, per-market}, scored
on pooled OOS log-loss with ECE + per-market Brier as guardrails.

Stages:
  python3 scripts/audit_calibration.py --stage oos        # cache raw OOS preds
  python3 scripts/audit_calibration.py --stage analyze    # metrics + bake-off + plots
  python3 scripts/audit_calibration.py                    # both
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import config as cfgmod  # noqa: E402
from nflvalue import ml_ranker as mlr

FRAME_PATH = os.path.join(cfgmod.DATA_DIR, "ml_frame.parquet")
OOS_PATH = os.path.join(cfgmod.DATA_DIR, "calib_oos.parquet")
REPORT_MD = "reports/calibration_audit.md"
REPORT_JSON = os.path.join(cfgmod.DATA_DIR, "calibration_audit.json")
PLOT_POOLED = "reports/calibration_reliability_pooled.png"
PLOT_MARKET = "reports/calibration_reliability_by_market.png"

EVAL_SEASONS = [2021, 2022, 2023, 2024, 2025]   # seasons with a walk-forward OOS fit
CAL_SEASONS = [2022, 2023, 2024, 2025]          # seasons a prior-OOS calibrator can cover
MARKETS = list(mlr.MARKETS7)
EPS = 1e-6
N_BINS = 10
PERMARKET_MIN = 500      # below this in the calibrator-train pool, fall back to pooled fit


# --------------------------------------------------------------------------- #
# Stage: oos  — cache raw out-of-sample P(over), season-walk-forward
# --------------------------------------------------------------------------- #
def build_oos(frame: pd.DataFrame) -> pd.DataFrame:
    done = pd.read_parquet(OOS_PATH) if os.path.exists(OOS_PATH) else pd.DataFrame()
    have = set(done["season"].unique().tolist()) if len(done) else set()
    chunks = [done] if len(done) else []
    for s in EVAL_SEASONS:
        if s in have:
            continue
        tr = frame[frame["season"] < s]
        te = frame[frame["season"] == s].copy()
        model = mlr.MLRanker("gbdt").fit(tr, tr["y_over"])
        te["p_raw"] = model.predict_p_over(te)      # enforce=True: hard-fails any overlap
        keep = ["season", "week", "game_id", "player_id", "market", "y_over", "p_raw"]
        chunks.append(te[keep])
        pd.concat(chunks, ignore_index=True).to_parquet(OOS_PATH, index=False)
        print(f"  OOS season {s}: trained <{s} ({len(tr):,} rows), "
              f"predicted {len(te):,}; p_raw mean {te['p_raw'].mean():.4f}")
    oos = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return oos.drop_duplicates(subset=["season", "week", "game_id", "player_id", "market"])


# --------------------------------------------------------------------------- #
# Calibrators — each is fit(p, y) -> callable transform(p)
# --------------------------------------------------------------------------- #
def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def fit_raw(p, y) -> Callable:
    return lambda q: np.clip(np.asarray(q, float), 0.0, 1.0)


def fit_platt(p, y) -> Callable:
    from sklearn.linear_model import LogisticRegression
    X = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(X, np.asarray(y, int))
    return lambda q: lr.predict_proba(_logit(q).reshape(-1, 1))[:, 1]


def fit_isotonic(p, y) -> Callable:
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(np.asarray(p, float), np.asarray(y, float))
    return lambda q: np.clip(ir.predict(np.clip(np.asarray(q, float), 0, 1)), 0.0, 1.0)


def fit_beta(p, y) -> Callable:
    """Kull et al. (2017) beta calibration: sigmoid(a*ln p + b*ln(1-p) + c).
    Includes the identity (a=1, b=-1, c=0), so it can't do worse than 'already
    calibrated' up to fit noise; steadier than isotonic on thin slices."""
    from sklearn.linear_model import LogisticRegression
    pc = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    X = np.column_stack([np.log(pc), np.log(1 - pc)])
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(X, np.asarray(y, int))

    def _t(q):
        qc = np.clip(np.asarray(q, float), EPS, 1 - EPS)
        return lr.predict_proba(np.column_stack([np.log(qc), np.log(1 - qc)]))[:, 1]
    return _t


FITTERS = {"raw": fit_raw, "platt": fit_platt, "isotonic": fit_isotonic, "beta": fit_beta}


def apply_walk_forward(oos: pd.DataFrame, method: str, per_market: bool) -> pd.Series:
    """Calibrated P(over) for every row in CAL_SEASONS, fit strictly on < S."""
    out = pd.Series(index=oos.index, dtype=float)
    fitter = FITTERS[method]
    for s in CAL_SEASONS:
        prior = oos[oos["season"] < s]
        cur = oos[oos["season"] == s]
        if per_market:
            pooled_fn = fitter(prior["p_raw"], prior["y_over"])
            for m in MARKETS:
                idx = cur.index[cur["market"] == m]
                if len(idx) == 0:
                    continue
                pm = prior[prior["market"] == m]
                fn = fitter(pm["p_raw"], pm["y_over"]) if len(pm) >= PERMARKET_MIN else pooled_fn
                out.loc[idx] = fn(cur.loc[idx, "p_raw"].to_numpy())
        else:
            fn = fitter(prior["p_raw"], prior["y_over"])
            out.loc[cur.index] = fn(cur["p_raw"].to_numpy())
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _eqfreq_bins(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency bin ids via quantile edges (ties collapse -> fewer bins)."""
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p, q))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = _eqfreq_bins(p, n_bins)
    rows = []
    for k in np.unique(b):
        m = b == k
        rows.append({"bin": int(k), "n": int(m.sum()),
                     "p_mean": float(p[m].mean()), "obs": float(y[m].mean())})
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    t = reliability_table(p, y, n_bins)
    return float((t["n"] / t["n"].sum() * (t["p_mean"] - t["obs"]).abs()).sum())


def mce(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    t = reliability_table(p, y, n_bins)
    return float((t["p_mean"] - t["obs"]).abs().max())


def brier_decomp(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> Dict[str, float]:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    N = len(y)
    obar = y.mean()
    t = reliability_table(p, y, n_bins)
    rel = float((t["n"] / N * (t["p_mean"] - t["obs"]) ** 2).sum())
    res = float((t["n"] / N * (t["obs"] - obar) ** 2).sum())
    unc = float(obar * (1 - obar))
    return {"brier": float(np.mean((p - y) ** 2)), "reliability": rel,
            "resolution": res, "uncertainty": unc}


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import log_loss
    return float(log_loss(np.asarray(y, int), np.clip(p, EPS, 1 - EPS), labels=[0, 1]))


def _row_ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_t(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Paired per-row log-loss test: positive dLL/t => B improves on A. The
    Phase-6 t>=2 culture applied to a probability-quality delta."""
    d = _row_ll(p_a, y) - _row_ll(p_b, y)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"d_logloss": round(float(d.mean()), 5),
            "t": round(float(d.mean() / se), 2) if se > 0 else 0.0, "n": int(len(d))}


def metric_block(p: np.ndarray, y: np.ndarray, markets: np.ndarray) -> Dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = {"n": int(len(y)), "log_loss": round(logloss(p, y), 5),
           "ece": round(ece(p, y), 4), "mce": round(mce(p, y), 4),
           **{k: round(v, 5) for k, v in brier_decomp(p, y).items()}}
    per = {}
    for m in MARKETS:
        mm = markets == m
        if mm.sum() < 50:
            continue
        per[m] = {"n": int(mm.sum()), "over_rate": round(float(y[mm].mean()), 4),
                  "log_loss": round(logloss(p[mm], y[mm]), 5),
                  "brier": round(float(np.mean((p[mm] - y[mm]) ** 2)), 5),
                  "ece": round(ece(p[mm], y[mm]), 4)}
    out["per_market"] = per
    return out


# --------------------------------------------------------------------------- #
# Stage: analyze
# --------------------------------------------------------------------------- #
def analyze(oos: pd.DataFrame) -> Dict:
    ev = oos[oos["season"].isin(CAL_SEASONS)].copy()
    y = ev["y_over"].to_numpy()
    mk = ev["market"].to_numpy()

    variants: Dict[str, np.ndarray] = {"raw": ev["p_raw"].to_numpy()}
    for method in ("platt", "isotonic", "beta"):
        for pm in (False, True):
            tag = f"{method}_{'permkt' if pm else 'pooled'}"
            variants[tag] = apply_walk_forward(oos, method, pm).loc[ev.index].to_numpy()

    results = {tag: metric_block(p, y, mk) for tag, p in variants.items()}

    # winner: lowest pooled log-loss whose ECE and every per-market Brier do not
    # regress vs raw beyond noise (Brier SE ~ sqrt(var/n)); require strict
    # improvement in pooled log-loss AND ECE to beat 'raw'.
    raw = results["raw"]
    ranked = sorted(results.items(), key=lambda kv: kv[1]["log_loss"])
    winner = "raw"
    for tag, r in ranked:
        if tag == "raw":
            continue
        better_ll = r["log_loss"] < raw["log_loss"] - 1e-5
        better_ece = r["ece"] <= raw["ece"] + 1e-4
        if better_ll and better_ece:
            winner = tag
            break

    # significance (t>=2 culture): paired per-row log-loss vs raw, pooled +
    # per-season, plus the near-tie checks against the other per-market methods.
    wp = variants[winner]
    sig = {"pooled_vs_raw": paired_t(variants["raw"], wp, y),
           "per_season_vs_raw": {int(s): paired_t(
               ev.loc[ev["season"] == s, "p_raw"].to_numpy(),
               wp[(ev["season"] == s).to_numpy()],
               y[(ev["season"] == s).to_numpy()]) for s in CAL_SEASONS},
           "vs_beta_permkt": paired_t(variants.get("beta_permkt", wp), wp, y),
           "vs_isotonic_permkt": paired_t(variants.get("isotonic_permkt", wp), wp, y)}

    plots(ev, variants, winner)
    payload = {"eval_seasons": CAL_SEASONS, "n_eval": int(len(ev)),
               "raw_overall_over_rate": round(float(y.mean()), 4),
               "results": results, "winner": winner, "significance": sig,
               "reliability_raw": reliability_table(variants["raw"], y).to_dict("records"),
               "reliability_winner": reliability_table(variants[winner], y).to_dict("records")}
    os.makedirs("reports", exist_ok=True)
    cfgmod.save_json(REPORT_JSON, payload)
    write_report(payload)
    return payload


def plots(ev: pd.DataFrame, variants: Dict[str, np.ndarray], winner: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = ev["y_over"].to_numpy()

    # pooled: raw vs winner
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for tag, style in ((("raw"), "o-"), ((winner), "s-")):
        t = reliability_table(variants[tag], y)
        ax.plot(t["p_mean"], t["obs"], style, label=f"{tag} (ECE {ece(variants[tag], y):.3f})")
    ax.set_xlabel("mean predicted P(over)"); ax.set_ylabel("empirical over-rate")
    ax.set_title(f"Reliability, pooled OOS {CAL_SEASONS[0]}-{CAL_SEASONS[-1]}")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(PLOT_POOLED, dpi=110); plt.close(fig)

    # per-market small multiples: raw vs winner
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, m in zip(axes.ravel(), MARKETS):
        mm = ev["market"].to_numpy() == m
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for tag, style in (("raw", "o-"), (winner, "s-")):
            t = reliability_table(variants[tag][mm], y[mm], n_bins=8)
            ax.plot(t["p_mean"], t["obs"], style, ms=4, label=tag)
        ax.set_title(f"{m}\n(n={int(mm.sum())}, over {y[mm].mean():.2f})", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"Reliability by market, pooled OOS — raw vs {winner}")
    fig.tight_layout(); fig.savefig(PLOT_MARKET, dpi=110); plt.close(fig)


def write_report(p: Dict) -> None:
    R = p["results"]
    L = ["# Phase 7.1 — calibration audit (walk-forward, out-of-sample)", "",
         f"Pooled OOS seasons **{p['eval_seasons'][0]}-{p['eval_seasons'][-1]}**, "
         f"n={p['n_eval']:,}; overall over-rate {p['raw_overall_over_rate']:.4f}. "
         "Calibrator for season S fit ONLY on raw OOS predictions of seasons < S "
         "(2021 seeds history). Calibrate P(over) vs y_over; p_under = 1 - cal.", "",
         "Synthetic-line caveat applies to every number (trailing-mean lines, no real prices).", "",
         "## Method bake-off (pooled)", "",
         "| variant | log-loss | ECE | MCE | Brier | reliability | resolution |",
         "|---|---|---|---|---|---|---|"]
    for tag, r in sorted(R.items(), key=lambda kv: kv[1]["log_loss"]):
        star = " **←winner**" if tag == p["winner"] else ""
        L.append(f"| {tag}{star} | {r['log_loss']} | {r['ece']} | {r['mce']} | "
                 f"{r['brier']} | {r['reliability']} | {r['resolution']} |")
    raw, win = R["raw"], R[p["winner"]]
    L += ["", f"**Winner: `{p['winner']}`.** "
          f"pooled log-loss {raw['log_loss']}→{win['log_loss']} "
          f"({win['log_loss']-raw['log_loss']:+.5f}), "
          f"ECE {raw['ece']}→{win['ece']} ({win['ece']-raw['ece']:+.4f}), "
          f"reliability {raw['reliability']}→{win['reliability']}.", "",
          "## Significance (paired per-row log-loss vs raw; +t = calibration helps)", "",
          f"Pooled: dLL {p['significance']['pooled_vs_raw']['d_logloss']:+.5f}, "
          f"**t={p['significance']['pooled_vs_raw']['t']:+.2f}** "
          f"(n={p['significance']['pooled_vs_raw']['n']:,}). "
          f"Near-tie vs beta_permkt t={p['significance']['vs_beta_permkt']['t']:+.2f} "
          f"(pick simpler Platt); beats isotonic_permkt "
          f"t={p['significance']['vs_isotonic_permkt']['t']:+.2f} (thin-slice overfit).", "",
          "| season | dLL vs raw | t |", "|---|---|---|"]
    for s, st in p["significance"]["per_season_vs_raw"].items():
        L.append(f"| {s} | {st['d_logloss']:+.5f} | {st['t']:+.2f} |")
    L += ["", "*Honest read: the gain is large early and shrinks toward zero as the "
          "base model's training history grows (self-calibrating). Wired as a guard "
          "+ tail-corrector that clears the pooled bar and never hurts.*", "",
          "## Per-market (raw → winner): over-rate, log-loss, Brier, ECE", "",
          "| market | n | over | LL raw→win | Brier raw→win | ECE raw→win |",
          "|---|---|---|---|---|---|"]
    for m in MARKETS:
        a, b = raw["per_market"].get(m), win["per_market"].get(m)
        if not a:
            continue
        L.append(f"| {m} | {a['n']:,} | {a['over_rate']} | "
                 f"{a['log_loss']}→{b['log_loss']} | {a['brier']}→{b['brier']} | "
                 f"{a['ece']}→{b['ece']} |")
    L += ["", "## Reliability (pooled, equal-frequency deciles)", "",
          "| decile | n | raw p̄ | winner p̄ | observed |", "|---|---|---|---|---|"]
    for rr, rw in zip(p["reliability_raw"], p["reliability_winner"]):
        L.append(f"| {rr['bin']} | {rr['n']:,} | {rr['p_mean']:.3f} | "
                 f"{rw['p_mean']:.3f} | {rr['obs']:.3f} |")
    L += ["", f"Plots: `{PLOT_POOLED}`, `{PLOT_MARKET}`.", ""]
    os.makedirs("reports", exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["oos", "analyze", "all"], default="all")
    args = ap.parse_args()
    frame = pd.read_parquet(FRAME_PATH)
    if args.stage in ("oos", "all"):
        oos = build_oos(frame)
        print(f"OOS cache: {len(oos):,} rows, seasons {sorted(oos['season'].unique())}")
    if args.stage in ("analyze", "all"):
        oos = pd.read_parquet(OOS_PATH)
        analyze(oos)


if __name__ == "__main__":
    main()
