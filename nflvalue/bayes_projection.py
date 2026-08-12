"""Challenger A — hierarchical Bayesian projection layer (pre-registered
2026-07-30, BUILD_PROMPTS_model_challengers_2026-07.md §A).

Partial pooling: per market, a player-level random effect is pooled through
position-role and team-offense levels on BOTH location and scale, so a
2-observation player borrows strength from his role/team and carries a wider
predictive than a 100-observation player at equal role.  Every covariate is
strictly pregame (the incumbent walk-forward projected mean, itself built
from shift(1)-then-roll features), and every fit is walk-forward: the model
for eval season S sees ONLY seasons < S.

DECLARED BEFORE THE EVAL RUN (per §A2.1 "declare which BEFORE the eval run
and keep it fixed across seasons"):

* Inference = SVI/ADVI (numpyro ``AutoNormal`` guide, Adam lr 0.01,
  ``SVI_STEPS`` full-batch steps).  NUTS would exceed the CPU budget for
  48 walk-forward fits (6 markets x 4 seasons x 2 seeds); ADVI is the
  registered choice and is NOT revisited after seeing results.
* Likelihoods: yards markets (receiving/rushing/passing_yards) are modeled
  as Normal on log1p(actual clipped at 0) with hierarchical location AND
  log-scale; count markets (receptions, pass/rush_attempts) as
  Gamma-Poisson (negative binomial) with hierarchical log-mean and
  role-pooled concentration.  anytime_td is OUT OF SCOPE (§A3: TD/exact
  markets stay fail-closed regardless of CRPS).
* Predictive = ``N_PRED_SAMPLES`` draws per row, propagating guide
  posterior uncertainty (posterior latent draws -> one observation draw
  each), seeded deterministically per (seed, market, season).
* Unseen-in-training players draw their random effects from the fitted
  prior (role/team pooled mean, tau-width) — the principled cold-start.

Fail-closed integration contract (§A2.4): the live path consumes this layer
ONLY behind config ``projection.bayes`` (default false) and only when a
validated artifact exists; absent/corrupt artifact -> bit-identical
incumbent behavior.  The gate FAILED (book/bayes_projection_eval.json), so
no consumer flag was ever added: this module is research machinery kept for
the registry, exactly like every other measured rejection.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np

YARD_MARKETS = ("receiving_yards", "rushing_yards", "passing_yards")
COUNT_MARKETS = ("receptions", "pass_attempts", "rush_attempts")
MARKETS_IN_SCOPE = YARD_MARKETS + COUNT_MARKETS

ROLES = ("QB", "RB", "WR", "TE")

SVI_STEPS = 2000
SVI_LR = 0.01
N_PRED_SAMPLES = 512


def _import_numpyro():
    import jax
    jax.config.update("jax_platform_name", "cpu")
    import numpyro
    numpyro.set_host_device_count(1)
    return numpyro


def _design(rows, role_index: Dict[str, int], team_index: Dict[str, int],
            player_index: Dict[str, int]):
    """rows -> integer index arrays + centered covariate.  ``rows`` is a
    DataFrame with player_id, role, team, mean_pred columns; player ids
    absent from ``player_index`` get -1 (unseen -> prior draw)."""
    x = np.log1p(np.clip(rows["mean_pred"].to_numpy(dtype=float), 0.0, None))
    role = rows["role"].map(role_index).to_numpy(dtype=int)
    team = rows["team"].map(lambda t: team_index.get(t, 0)).to_numpy(dtype=int)
    player = rows["player_id"].map(
        lambda p: player_index.get(p, -1)).to_numpy(dtype=int)
    return x, role, team, player


def _model_factory(family: str, n_role: int, n_team: int, n_player: int,
                   x_center: float):
    """Build the numpyro model fn for one market family."""
    import numpyro
    import numpyro.distributions as dist
    import jax.numpy as jnp

    def model(x, role, team, player, y=None):
        a = numpyro.sample("a", dist.Normal(0.0, 2.0))
        beta = numpyro.sample("beta", dist.Normal(1.0, 1.0))
        c_role = numpyro.sample("c_role", dist.Normal(0.0, 1.0).expand([n_role]))
        tau_t = numpyro.sample("tau_t", dist.HalfNormal(0.5))
        d_team = numpyro.sample(
            "d_team", dist.Normal(0.0, 1.0).expand([n_team])) * tau_t
        tau_p = numpyro.sample("tau_p", dist.HalfNormal(0.5))
        b_raw = numpyro.sample("b_raw", dist.Normal(0.0, 1.0).expand([n_player]))
        b_player = b_raw * tau_p
        xc = x - x_center
        loc = a + beta * xc + c_role[role] + d_team[team] + b_player[player]
        if family == "yards":
            s0 = numpyro.sample("s0", dist.Normal(-0.5, 1.0))
            s_role = numpyro.sample("s_role", dist.Normal(0.0, 0.5).expand([n_role]))
            tau_s = numpyro.sample("tau_s", dist.HalfNormal(0.3))
            s_raw = numpyro.sample("s_raw", dist.Normal(0.0, 1.0).expand([n_player]))
            log_sigma = s0 + s_role[role] + (s_raw * tau_s)[player]
            sigma = jnp.exp(log_sigma)
            numpyro.sample("y", dist.Normal(loc, sigma), obs=y)
        else:
            q0 = numpyro.sample("q0", dist.Normal(1.0, 1.0))
            q_role = numpyro.sample("q_role", dist.Normal(0.0, 0.5).expand([n_role]))
            phi = jnp.exp(q0 + q_role[role])
            mean = jnp.exp(loc)
            numpyro.sample("y", dist.GammaPoisson(phi, phi / mean), obs=y)

    return model


class BayesProjection:
    """One fitted market-model: walk-forward, seasons < eval season only."""

    def __init__(self, market: str, seed: int):
        if market not in MARKETS_IN_SCOPE:
            raise ValueError(
                f"market {market!r} out of scope; TD/exact stays fail-closed")
        self.market = market
        self.family = "yards" if market in YARD_MARKETS else "counts"
        self.seed = int(seed)
        self.posterior: Optional[Dict[str, np.ndarray]] = None
        self.meta: Dict = {}
        self.role_index: Dict[str, int] = {}
        self.team_index: Dict[str, int] = {}
        self.player_index: Dict[str, int] = {}
        self.x_center: float = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, train_rows, eval_season: int) -> "BayesProjection":
        """``train_rows``: DataFrame with player_id, role, team, mean_pred,
        actual, season, week — every season strictly < ``eval_season``
        (asserted; leakage here is the one unforgivable bug)."""
        bad = train_rows[train_rows["season"] >= eval_season]
        if len(bad):
            raise ValueError(
                f"walk-forward violation: {len(bad)} training rows at/after "
                f"eval season {eval_season}")
        _import_numpyro()
        import jax
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal
        from numpyro.optim import Adam

        rows = train_rows.dropna(subset=["mean_pred", "actual"]).copy()
        rows = rows.sort_values(["player_id", "season", "week"], kind="mergesort")
        self.role_index = {r: i for i, r in enumerate(ROLES)}
        teams = sorted(rows["team"].astype(str).unique().tolist())
        self.team_index = {t: i for i, t in enumerate(teams)}
        players = sorted(rows["player_id"].astype(str).unique().tolist())
        self.player_index = {p: i for i, p in enumerate(players)}
        x, role, team, player = _design(rows, self.role_index, self.team_index,
                                        self.player_index)
        self.x_center = float(np.mean(x))
        y_raw = rows["actual"].to_numpy(dtype=float)
        if self.family == "yards":
            y = np.log1p(np.clip(y_raw, 0.0, None))
        else:
            y = np.clip(np.round(y_raw), 0, None).astype(np.int32)

        model = _model_factory(self.family, len(ROLES), len(teams), len(players),
                               self.x_center)
        guide = AutoNormal(model)
        svi = SVI(model, guide, Adam(SVI_LR), Trace_ELBO())
        rng = jax.random.PRNGKey(self.seed * 1_000_003 + eval_season * 101
                                 + _stable_hash(self.market) % 97)
        import jax.numpy as jnp
        result = svi.run(rng, SVI_STEPS, jnp.asarray(x), jnp.asarray(role),
                         jnp.asarray(team), jnp.asarray(player), y=jnp.asarray(y),
                         progress_bar=False)
        params = result.params
        # posterior latent draws from the fitted AutoNormal guide,
        # deterministic key -> N_PRED_SAMPLES x latent arrays (numpy)
        from numpyro.infer import Predictive
        pred = Predictive(guide, params=params, num_samples=N_PRED_SAMPLES)
        key = jax.random.PRNGKey(self.seed * 7_777_777 + eval_season)
        latents = pred(key, jnp.asarray(x[:1]), jnp.asarray(role[:1]),
                       jnp.asarray(team[:1]), jnp.asarray(player[:1]))
        self.posterior = {k: np.asarray(v) for k, v in latents.items()}
        self.meta = {
            "market": self.market, "family": self.family, "seed": self.seed,
            "eval_season": int(eval_season),
            "train_seasons": sorted(int(s) for s in rows["season"].unique()),
            "n_train_rows": int(len(rows)), "n_players": len(players),
            "inference": f"SVI/ADVI AutoNormal, Adam lr {SVI_LR}, {SVI_STEPS} steps",
            "n_pred_samples": N_PRED_SAMPLES,
        }
        return self

    # ------------------------------------------------------- predictive draws
    def predictive_samples(self, rows) -> np.ndarray:
        """(n_rows, N_PRED_SAMPLES) array of predictive draws on the ORIGINAL
        stat scale, propagating posterior uncertainty.  Unseen players draw
        their effects from the fitted prior (deterministic per-player eps)."""
        if self.posterior is None:
            raise RuntimeError("not fitted")
        post = self.posterior
        x, role, team, player = _design(rows, self.role_index, self.team_index,
                                        self.player_index)
        S = N_PRED_SAMPLES
        n = len(x)
        xc = x - self.x_center

        a = post["a"].reshape(S, 1)
        beta = post["beta"].reshape(S, 1)
        c_role = post["c_role"][:, role]
        d_team = post["d_team"][:, team] * post["tau_t"].reshape(S, 1)
        tau_p = post["tau_p"].reshape(S, 1)
        b = np.where(player[None, :] >= 0,
                     post["b_raw"][:, np.clip(player, 0, None)], 0.0) * tau_p
        # unseen players: prior draw, one deterministic eps per (player, sample)
        unseen = player < 0
        if unseen.any():
            eps = _det_normal(rows["player_id"].to_numpy()[unseen], S,
                              tag=f"{self.market}-b")
            b[:, unseen] = eps.T * tau_p
        loc = a + beta * xc[None, :] + c_role + d_team + b

        tag = f"pred-{self.market}-{self.meta.get('eval_season')}-{self.seed}"
        rng = np.random.default_rng(_stable_hash(tag) % (2 ** 32))
        if self.family == "yards":
            s0 = post["s0"].reshape(S, 1)
            s_role = post["s_role"][:, role]
            tau_s = post["tau_s"].reshape(S, 1)
            sp = np.where(player[None, :] >= 0,
                          post["s_raw"][:, np.clip(player, 0, None)], 0.0) * tau_s
            if unseen.any():
                eps_s = _det_normal(rows["player_id"].to_numpy()[unseen], S,
                                    tag=f"{self.market}-s")
                sp[:, unseen] = eps_s.T * tau_s
            sigma = np.exp(s0 + s_role + sp)
            z = loc + sigma * rng.standard_normal(size=(S, n))
            # Numerical hygiene, decided before any eval numbers were read:
            # an extreme posterior-tail z overflows expm1 to inf, which turns
            # a (deservedly) terrible CRPS into NaN and poisons the pooled
            # mean.  Cap the log-scale draw at log(1e6): a 1,000,000-yard
            # draw still pays an enormous, finite CRPS penalty.
            z = np.minimum(z, np.log(1e6))
            draws = np.expm1(z)
            return np.clip(draws.T, 0.0, None)
        q0 = post["q0"].reshape(S, 1)
        q_role = post["q_role"][:, role]
        phi = np.exp(np.clip(q0 + q_role, -20.0, 20.0))
        mean = np.exp(np.minimum(loc, np.log(1e6)))   # same anti-overflow cap
        lam = rng.gamma(shape=phi, scale=mean / phi)
        draws = rng.poisson(lam=np.minimum(lam, 1e6)).astype(float)
        return draws.T

    # ------------------------------------------------------------- persistence
    def save(self, path: str) -> str:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"market": self.market, "seed": self.seed,
                     "posterior": self.posterior, "meta": self.meta,
                     "role_index": self.role_index, "team_index": self.team_index,
                     "player_index": self.player_index, "x_center": self.x_center},
                    path, compress=3)
        with open(path + ".sha256", "w") as fh:
            fh.write(_file_sha256(path) + "\n")
        return path

    @classmethod
    def load(cls, path: str) -> "BayesProjection":
        """Fail closed: missing or digest-mismatched artifact raises, and the
        caller's contract is to fall back to the incumbent path untouched."""
        import joblib
        sidecar = path + ".sha256"
        if os.path.exists(sidecar):
            with open(sidecar) as fh:
                expected = fh.read().strip()
            if _file_sha256(path) != expected:
                raise ValueError(f"bayes artifact {path} failed integrity check")
        blob = joblib.load(path)
        obj = cls(blob["market"], blob["seed"])
        obj.posterior = blob["posterior"]
        obj.meta = blob["meta"]
        obj.role_index = blob["role_index"]
        obj.team_index = blob["team_index"]
        obj.player_index = blob["player_index"]
        obj.x_center = blob["x_center"]
        return obj


# ------------------------------------------------------------------ helpers
def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:12], 16)


def _det_normal(player_ids, n_samples: int, tag: str) -> np.ndarray:
    """(n_players, n_samples) standard normals, deterministic per player id —
    an unseen player's prior draw must not depend on row order."""
    out = np.empty((len(player_ids), n_samples))
    for i, pid in enumerate(player_ids):
        rng = np.random.default_rng(_stable_hash(f"{tag}-{pid}") % (2 ** 32))
        out[i] = rng.standard_normal(n_samples)
    return out


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def crps_from_samples(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Sample-based CRPS, one value per row.  ``samples`` is (n_rows, S).
    CRPS = E|X - y| - 0.5 E|X - X'| via the sorted-sample identity.
    The SAME estimator scores both arms (declared; no closed-form asymmetry)."""
    n, S = samples.shape
    term1 = np.abs(samples - actual[:, None]).mean(axis=1)
    srt = np.sort(samples, axis=1)
    w = 2.0 * np.arange(1, S + 1) - S - 1.0        # sorted-sample E|X-X'| weights
    term2 = (srt * w[None, :]).sum(axis=1) / (S * S)
    return term1 - term2


def incumbent_samples(mean: np.ndarray, sd: np.ndarray, dists: List[str],
                      seed: int, n_samples: int = N_PRED_SAMPLES) -> np.ndarray:
    """Predictive draws from the INCUMBENT parametric family per row, using
    the exact parameterizations of nflvalue.projection's survival helpers."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    n = len(mean)
    out = np.empty((n, n_samples))
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    darr = np.asarray(dists)
    for dist_name in np.unique(darr):
        m = darr == dist_name
        mu = np.clip(mean[m], 1e-6, None)
        s = np.clip(sd[m], 1e-6, None)
        if dist_name == "normal":
            out[m] = (mean[m][:, None]
                      + s[:, None] * rng.standard_normal((m.sum(), n_samples)))
        elif dist_name == "gamma":
            shape = (mu / s) ** 2
            scale = (s ** 2) / mu
            out[m] = stats.gamma.rvs(a=shape[:, None], scale=scale[:, None],
                                     size=(m.sum(), n_samples), random_state=rng)
        elif dist_name == "negbinom":
            var = np.maximum(s ** 2, mu * 1.01)
            p = mu / var
            r = mu ** 2 / (var - mu)
            out[m] = stats.nbinom.rvs(r[:, None], p[:, None],
                                      size=(m.sum(), n_samples), random_state=rng)
        elif dist_name == "poisson":
            out[m] = stats.poisson.rvs(mu[:, None], size=(m.sum(), n_samples),
                                       random_state=rng)
        else:
            raise ValueError(f"unknown incumbent dist {dist_name!r}")
    return out


def save_book(path: str, book: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(book, fh, indent=1, sort_keys=True)
