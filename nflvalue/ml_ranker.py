"""ML ranking layer: learn P(actual > line) on top of the deterministic model.

Architecture (the slot Phase 1 designed for): the deterministic projection
stays the source of every published NUMBER; the ML model is a STACKED
CLASSIFIER whose features include the deterministic model's own belief
(p_over, z) plus the walk-forward usage/efficiency/context features. Its
output is a probability used for RANKING (and, live, for the edge
comparison) — so the system keeps its auditability: projection explains the
number, the classifier explains the ordering.

Models (both seeded; GBDT is byte-reproducible, RF with n_jobs=-1 is
reproducible to one float ULP -- parallel vote averaging is order-dependent;
pass n_jobs=1 semantics via a single-thread environment if bitwise identity
ever matters more than the 2x fit speed):
  "gbdt"  HistGradientBoostingClassifier — gradient boosting minimizes
          log-loss by gradient steps; this is the "gradient descent score"
          being optimized. Handles NaNs natively. Default.
  "rf"    RandomForestClassifier — variance-reduction ensemble baseline
          (no gradient descent involved; run for comparison).

Anti-leakage is structural, not hopeful: ``fit`` records the latest
(season, week) it saw; ``predict`` REFUSES rows at or before that cutoff
unless they're strictly later... inverted: refuses rows unless every train
row predates every predict row (assert_walk_forward). Features are already
strictly-prior-week by construction (they come from the candidate frame).

Honesty rules carried over: evaluation is out-of-sample by season (or by
week for in-season retraining); the baseline it must beat is the TUNED
composite on the identical candidate pool; hit rates are at synthetic lines
(no free price history) with the 52.38% breakeven proxy; flag-gated OFF in
config until the evidence says otherwise ("ml_ranker" section).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEED = 20260701
MODEL_PATH_DEFAULT = "data/ml_ranker.joblib"

MARKETS7 = ("receiving_yards", "receptions", "rushing_yards", "passing_yards",
            "pass_attempts", "rush_attempts", "anytime_td")
POSITIONS = ("QB", "RB", "WR", "TE")

NUMERIC_FEATURES = [
    # deterministic model beliefs
    "p_over", "z", "mean", "sd", "line", "mean_minus_line", "sd_over_line",
    # projection components
    "opp_factor", "game_script", "proj_volume", "proj_efficiency",
    # walk-forward usage / efficiency (joined from player_week)
    "roll_games", "roll_targets", "roll_target_share", "roll_carries",
    "roll_carry_share", "roll_pass_attempts", "roll_adot", "roll_air_yards",
    "roll_ypt", "roll_catch_rate", "roll_ypc", "roll_ypa",
    # game context
    "team_margin", "total_line", "home", "week",
    # deterministic personal/defensive context (context_features.py) -- the
    # classifier decides their weight from outcomes; NaN = data unavailable
    # Birthday remains in the existing artifact schema for backward
    # compatibility, but the all-data audit found no positive effect. It is
    # not an enabled context multiplier and must not be described as an edge.
    "is_birthday_week", "revenge_game", "def_out_total", "def_out_db",
    "opp_epa_factor",
]

# advanced process metrics (advanced_features.py): strategic aggression,
# NGS, red-zone roles, O-line health, QB continuity, contract year, weather
from .advanced_features import FEATURES as _ADV_FEATURES  # noqa: E402
from .chemistry import FEATURES as _CHEM_FEATURES  # noqa: E402
from .ftn_features import FEATURES as _FTN_FEATURES  # noqa: E402
NUMERIC_FEATURES = NUMERIC_FEATURES + _ADV_FEATURES + _CHEM_FEATURES + _FTN_FEATURES


def build_features(cands: pd.DataFrame, pw: pd.DataFrame,
                   pack=None, adv=None) -> pd.DataFrame:
    """Candidate frame + player_week join -> model-ready feature frame.

    Every input column is walk-forward by construction (candidate rows carry
    strictly-prior-week features; the pw join brings the SAME week's roll_*
    columns, which are also prior-week by features.py's shift-then-roll).
    ``pack`` (context_features.ContextPack) adds birthday/revenge/defensive-
    injury/opp-EPA features; ``adv`` (advanced_features.AdvancedPack) adds
    the process metrics. Either None stamps neutral values."""
    from .advanced_features import attach_neutral
    from .chemistry import attach_neutral as chem_neutral
    from .context_features import attach
    f = attach(cands, pack)
    f = adv.attach(f) if adv is not None else attach_neutral(f)
    if not all(c in f.columns for c in _CHEM_FEATURES):
        f = chem_neutral(f)   # chemistry stamped upstream when its pack exists
    if not all(c in f.columns for c in _FTN_FEATURES):
        from .ftn_features import attach_neutral as ftn_neutral
        f = ftn_neutral(f)
    comps = f["components"].apply(lambda c: c or {})
    f["opp_factor"] = comps.apply(lambda c: c.get("opp_factor", 1.0)).astype(float)
    f["game_script"] = comps.apply(lambda c: c.get("game_script", 1.0)).astype(float)
    f["proj_volume"] = comps.apply(lambda c: c.get("volume", np.nan)).astype(float)
    f["proj_efficiency"] = comps.apply(lambda c: c.get("efficiency", np.nan)).astype(float)

    f["z"] = (f["mean"] - f["line"]) / f["sd"].clip(lower=1e-6)
    f["mean_minus_line"] = f["mean"] - f["line"]
    f["sd_over_line"] = f["sd"] / f["line"].abs().clip(lower=1.0)
    f["team_margin"] = np.where(f["home"].astype(bool),
                                f["spread_line"].astype(float),
                                -f["spread_line"].astype(float))
    f["home"] = f["home"].astype(int)

    roll_cols = ["roll_games", "roll_targets", "roll_target_share", "roll_carries",
                 "roll_carry_share", "roll_pass_attempts", "roll_adot", "roll_air_yards",
                 "roll_ypt", "roll_catch_rate", "roll_ypc", "roll_ypa"]
    pw_slim = pw[["season", "week", "player_id"] + roll_cols].drop_duplicates(
        subset=["season", "week", "player_id"])
    f = f.drop(columns=[c for c in roll_cols if c in f.columns], errors="ignore")
    f = f.merge(pw_slim, on=["season", "week", "player_id"], how="left")

    for m in MARKETS7:
        f[f"mkt_{m}"] = (f["market"] == m).astype(int)
    for p in POSITIONS:
        f[f"pos_{p}"] = (f["pos"] == p).astype(int)
    return f


def feature_columns() -> List[str]:
    return NUMERIC_FEATURES + [f"mkt_{m}" for m in MARKETS7] + [f"pos_{p}" for p in POSITIONS]


def label_over(frame: pd.DataFrame, actuals: Dict[tuple, float]) -> pd.Series:
    """y = 1 if the actual landed OVER the line (anytime_td: scored)."""
    y = []
    for r in frame.itertuples(index=False):
        a = actuals.get((r.player_id, r.market))
        if a is None:
            y.append(np.nan)
        elif r.market == "anytime_td":
            y.append(1.0 if a >= 1.0 else 0.0)
        else:
            y.append(1.0 if a > r.line else 0.0)
    return pd.Series(y, index=frame.index)


class WalkForwardViolation(RuntimeError):
    """Train data at/after predict data -- the one unforgivable ML bug here."""


class MLRanker:
    def __init__(self, model: str = "gbdt", seed: int = SEED, **kw):
        self.model_name = model
        self.seed = seed
        self.kw = kw
        self.clf = None
        self.train_max: Optional[Tuple[int, int]] = None

    def _new_clf(self):
        if self.model_name == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=int(self.kw.get("n_estimators", 400)),
                min_samples_leaf=int(self.kw.get("min_samples_leaf", 25)),
                max_features="sqrt", n_jobs=-1, random_state=self.seed)
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(self.kw.get("learning_rate", 0.06)),
            max_iter=int(self.kw.get("max_iter", 400)),
            max_leaf_nodes=int(self.kw.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(self.kw.get("min_samples_leaf", 40)),
            l2_regularization=float(self.kw.get("l2", 1.0)),
            early_stopping=True, validation_fraction=0.12,
            random_state=self.seed)

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "MLRanker":
        cols = feature_columns()
        mask = y.notna()
        X = frame.loc[mask, cols]
        if self.model_name == "rf":
            X = X.fillna(-999.0)          # RF can't take NaN; sentinel is fine for trees
        self.clf = self._new_clf().fit(X, y[mask].astype(int))
        self.train_max = (int(frame.loc[mask, "season"].max()),
                          int(frame.loc[mask].query("season == season.max()")["week"].max()))
        return self

    def assert_walk_forward(self, frame: pd.DataFrame) -> None:
        if self.train_max is None:
            raise WalkForwardViolation("model not fitted")
        s, w = self.train_max
        bad = frame[(frame["season"] < s)
                    | ((frame["season"] == s) & (frame["week"] <= w))]
        if len(bad):
            raise WalkForwardViolation(
                f"predict rows at/before train cutoff {self.train_max}: "
                f"{sorted(set(zip(bad['season'], bad['week'])))[:5]} ... -- "
                "an ML ranker may never score a week it trained on")

    def predict_p_over(self, frame: pd.DataFrame, enforce: bool = True) -> np.ndarray:
        if enforce:
            self.assert_walk_forward(frame)
        X = frame[feature_columns()]
        if self.model_name == "rf":
            X = X.fillna(-999.0)
        return self.clf.predict_proba(X)[:, 1]

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str = MODEL_PATH_DEFAULT) -> str:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model_name": self.model_name, "seed": self.seed,
                     "kw": self.kw, "clf": self.clf, "train_max": self.train_max}, path)
        return path

    @classmethod
    def load(cls, path: str = MODEL_PATH_DEFAULT) -> "MLRanker":
        import joblib
        blob = joblib.load(path)
        obj = cls(blob["model_name"], blob["seed"], **blob.get("kw", {}))
        obj.clf, obj.train_max = blob["clf"], tuple(blob["train_max"])
        return obj


# --------------------------------------------------------------------------- #
# Ranking with ML probabilities (same selection protocol as production)
# --------------------------------------------------------------------------- #
def rank_and_grade(frame: pd.DataFrame, p_over: np.ndarray,
                   top_n: int = 5, max_per_player: int = 2) -> pd.DataFrame:
    """Top-N per game by ML side-probability (yes-only markets stay yes),
    per-player capped, deterministic tie-breaks. Returns the graded leans."""
    f = frame.copy()
    f["ml_p_over"] = p_over
    yes_only = f["market"] == "anytime_td"
    f["ml_side"] = np.where(yes_only | (f["ml_p_over"] >= 0.5), "over", "under")
    f["ml_p_side"] = np.where(f["ml_side"] == "over", f["ml_p_over"], 1 - f["ml_p_over"])
    f["ml_hit"] = np.where(f["ml_side"] == "over", f["y_over"], 1 - f["y_over"])

    f = f.sort_values(["season", "week", "game_id", "ml_p_side", "player_id", "market"],
                      ascending=[True, True, True, False, True, True], kind="mergesort")
    keep_idx = []
    cur, taken, per_player = None, 0, {}
    for i, r in enumerate(f.itertuples(index=False)):
        g = (r.season, r.week, r.game_id)
        if g != cur:
            cur, taken, per_player = g, 0, {}
        if taken >= top_n or per_player.get(r.player_id, 0) >= max_per_player:
            continue
        per_player[r.player_id] = per_player.get(r.player_id, 0) + 1
        taken += 1
        keep_idx.append(i)
    return f.iloc[keep_idx].reset_index(drop=True)


def implied_units_at_110(hits: int, n: int) -> float:
    """P/L in flat 1u stakes if every lean were a real -110 price."""
    return round(hits * (100 / 110) - (n - hits), 2)
