#!/usr/bin/env python3
"""Bayesian factor weighting on the candidate frame.

1. Pooled Bayesian logistic regression (Gaussian prior = ridge MAP, Laplace
   posterior) -> coefficient posteriors: which factors have credible weight.
2. Hierarchical per-market partial pooling: market coefs shrunk toward the
   pooled posterior -> which factors deviate BY MARKET (weight differently).
3. Correlation clustering -> factor blocks that should be weighted TOGETHER.
"""
import json, sys, os
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nflvalue import ml_ranker as mlr

frame = pd.read_parquet("data/ml_frame.parquet")
NUM = [c for c in mlr.NUMERIC_FEATURES]
X0 = frame[NUM].astype(float)
med = X0.median()
X = ((X0.fillna(med) - X0.fillna(med).mean()) / (X0.fillna(med).std() + 1e-9)).to_numpy()
y = frame["y_over"].to_numpy(float)
mk = frame["market"].to_numpy()

def map_logistic(X, y, prior_mean, prior_prec, iters=60):
    """Newton MAP for logistic with Gaussian prior N(prior_mean, prior_prec^-1)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    pm = np.append(prior_mean, 0.0)
    P = np.diag(np.append(np.full(d, prior_prec) if np.isscalar(prior_prec) else prior_prec, 1e-4))
    b = pm.copy()
    for _ in range(iters):
        z = Xb @ b
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p)
        g = Xb.T @ (y - p) - P @ (b - pm)
        H = (Xb * W[:, None]).T @ Xb + P
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8: break
    cov = np.linalg.inv(H)
    return b, cov

# ---- pooled
prior_prec = 25.0     # N(0, 0.2^2) on standardized coefs: skeptical, market-is-sharp prior
b, cov = map_logistic(X, y, np.zeros(X.shape[1]), prior_prec)
se = np.sqrt(np.diag(cov))[:-1]
coefs = pd.DataFrame({"feature": NUM, "post_mean": b[:-1], "post_sd": se})
coefs["ci_lo"] = coefs.post_mean - 1.645 * coefs.post_sd
coefs["ci_hi"] = coefs.post_mean + 1.645 * coefs.post_sd
from scipy.stats import norm
coefs["P_gt0"] = 1 - norm.cdf(0, loc=coefs.post_mean, scale=coefs.post_sd)
coefs["credible"] = (coefs.ci_lo > 0) | (coefs.ci_hi < 0)
coefs = coefs.sort_values("post_mean", key=np.abs, ascending=False)

# ---- hierarchical per-market (partial pooling toward pooled posterior)
per_market = {}
tau_prec = 100.0      # market deviations shrunk hard toward pooled (tau=0.1)
for m in sorted(set(mk)):
    sel = mk == m
    if sel.sum() < 800: continue
    bm, covm = map_logistic(X[sel], y[sel], b[:-1], tau_prec)
    sem = np.sqrt(np.diag(covm))[:-1]
    dev = bm[:-1] - b[:-1]
    zdev = dev / np.sqrt(sem**2 + se**2)
    top = np.argsort(-np.abs(zdev))[:6]
    per_market[m] = [{"feature": NUM[i], "market_coef": round(float(bm[i]), 4),
                      "pooled_coef": round(float(b[i]), 4), "z_dev": round(float(zdev[i]), 2)}
                     for i in top if abs(zdev[i]) > 1.3]

# ---- correlation blocks (weight together)
C = np.corrcoef(X, rowvar=False)
D = 1 - np.abs(C)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
np.fill_diagonal(D, 0.0)
D = (D + D.T) / 2
L = linkage(squareform(D, checks=False), method="average")
labels = fcluster(L, t=0.55, criterion="distance")
blocks = {}
for f, l in zip(NUM, labels):
    blocks.setdefault(int(l), []).append(f)
blocks = {f"block_{k}": v for k, v in sorted(blocks.items(), key=lambda kv: -len(kv[1])) if len(v) > 1}

out = {
    "prior": "coef ~ N(0, 0.2^2) on standardized features (skeptical); Laplace posterior",
    "n": int(len(y)), "base_rate_y_over": round(float(np.nanmean(y)), 4),
    "credible_factors": coefs[coefs.credible][["feature", "post_mean", "post_sd", "ci_lo", "ci_hi"]]
        .round(4).to_dict("records"),
    "not_credible": coefs[~coefs.credible]["feature"].tolist(),
    "per_market_deviations": per_market,
    "weight_together_blocks": blocks,
}
json.dump(out, open("book/bayes_weights.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["credible_factors", "per_market_deviations"]}, indent=1)[:3500])
print("BLOCKS:", json.dumps(blocks, indent=1)[:1200])
