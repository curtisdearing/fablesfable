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

# Phase 7.1 calibration: below this many rows in a per-market calibrator-train
# slice, fall back to the pooled map (thin passing/TD slices can't support a
# stable per-market fit -- the 7.1 audit measured isotonic overfitting there).
CALIB_PERMARKET_MIN = 500
_CAL_EPS = 1e-6

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
    "is_birthday_week", "revenge_game", "def_out_total", "def_out_db",
    "opp_epa_factor",
    # Phase 6.1: player depth/location profiles + opponent red-zone defense
    "roll_short_tgt_share", "roll_mid_tgt_share", "roll_short_pass_share",
    "opp_rz_td_factor",
    # Phase 6.5: opponent-secondary absence factor + in-game durability
    "opp_absence_factor", "roll_early_exit_rate",
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
                 "roll_ypt", "roll_catch_rate", "roll_ypc", "roll_ypa",
                 "roll_short_tgt_share", "roll_mid_tgt_share", "roll_short_pass_share",
                 "roll_early_exit_rate"]
    # a caller's pw fixture may predate a later roll_* addition (Phase 6.1/6.5)
    # -- select only what exists and let the NUMERIC_FEATURES NaN-fill below
    # cover the rest, instead of a hard KeyError on an incomplete frame.
    have_roll = [c for c in roll_cols if c in pw.columns]
    pw_slim = pw[["season", "week", "player_id"] + have_roll].drop_duplicates(
        subset=["season", "week", "player_id"])
    f = f.drop(columns=[c for c in roll_cols if c in f.columns], errors="ignore")
    f = f.merge(pw_slim, on=["season", "week", "player_id"], how="left")

    for m in MARKETS7:
        f[f"mkt_{m}"] = (f["market"] == m).astype(int)
    for p in POSITIONS:
        f[f"pos_{p}"] = (f["pos"] == p).astype(int)
    # any numeric feature a caller's path didn't stamp exists as NaN (e.g.
    # opp_absence_factor when injury data is unavailable) -- GBDT handles NaN
    for c in NUMERIC_FEATURES:
        if c not in f.columns:
            f[c] = np.nan
    return f


def feature_columns(numeric: Optional[List[str]] = None) -> List[str]:
    """Full model-ready column list. ``numeric=None`` (default) uses every
    feature in NUMERIC_FEATURES -- pass a subset (Phase 7.2 walk-forward
    pruning, ``MLRanker(..., features=[...])``) to drop dead ones; the market
    and position dummies are structural and always included."""
    base = NUMERIC_FEATURES if numeric is None else list(numeric)
    return base + [f"mkt_{m}" for m in MARKETS7] + [f"pos_{p}" for p in POSITIONS]


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


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), _CAL_EPS, 1 - _CAL_EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Walk-forward probability calibrator for the ranker's P(over).

    Chosen from the Phase 7.1 audit (scripts/audit_calibration.py,
    docs/decisions_p7.md): **per-market Platt** — a 2-parameter logistic map on
    the logit of P(over), fit separately per market with a pooled fallback on
    thin slices. It won pooled out-of-sample log-loss (paired t=+7.2 vs the raw
    GBDT), tied beta calibration (t=1.4, so the simpler map is chosen), and beat
    per-market isotonic (t=+4.1; isotonic overfit the thin passing/TD slices).
    The raw GBDT is well-calibrated in aggregate but OVERCONFIDENT at the tails
    (top decile predicted .61, observed .53) — exactly the region edge and Kelly
    read — so the fix concentrates where it matters even when the pooled gain is
    modest, and shrinks toward zero as the base model's training history grows.

    Fit ONLY on out-of-sample base predictions (``MLRanker._fit_calibrator``
    generates them by expanding-season folds strictly before each fold), so the
    calibrator never sees a row it later corrects. Maps P(over) -> calibrated
    P(over); the caller sets p_under = 1 - calibrated exactly as before, so the
    composite edge interface (edge = calibrated P(side) - de-vigged market prob)
    is unchanged."""

    def __init__(self, method: str = "platt", per_market: bool = True,
                 permarket_min: int = CALIB_PERMARKET_MIN):
        self.method = method
        self.per_market = per_market
        self.permarket_min = permarket_min
        self.pooled = None
        self.by_market: Dict[str, object] = {}

    @staticmethod
    def _fit_map(p: np.ndarray, y: np.ndarray):
        # Platt scaling: logistic on logit(p). Weak regularization (C large) so
        # it is ~MLE yet steady on thin slices; the identity (A=1, B=0) is in
        # the family, so an already-calibrated market maps to ~itself.
        from sklearn.linear_model import LogisticRegression
        x = _logit(p).reshape(-1, 1)
        return LogisticRegression(C=1e6, solver="lbfgs").fit(x, np.asarray(y, int))

    @staticmethod
    def _apply_map(lr, p: np.ndarray) -> np.ndarray:
        return lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    def fit(self, p, y, markets) -> "Calibrator":
        p, y, markets = np.asarray(p, float), np.asarray(y, int), np.asarray(markets)
        self.pooled = self._fit_map(p, y)
        self.by_market = {}
        if self.per_market:
            for m in np.unique(markets):
                mask = markets == m
                if int(mask.sum()) >= self.permarket_min:
                    self.by_market[str(m)] = self._fit_map(p[mask], y[mask])
        return self

    def transform(self, p, markets) -> np.ndarray:
        p, markets = np.asarray(p, float), np.asarray(markets)
        out = self._apply_map(self.pooled, p)
        for m, lr in self.by_market.items():
            mask = markets == m
            if mask.any():
                out[mask] = self._apply_map(lr, p[mask])
        return np.clip(out, 0.0, 1.0)


class WalkForwardViolation(RuntimeError):
    """Train data at/after predict data -- the one unforgivable ML bug here."""


class MLRanker:
    def __init__(self, model: str = "gbdt", seed: int = SEED,
                 calibrate: Optional[str] = None,
                 features: Optional[List[str]] = None, **kw):
        self.model_name = model
        self.seed = seed
        self.calibrate = calibrate      # None | "platt_permkt" | "platt_pooled"
        # Phase 7.2 walk-forward feature pruning: None = every NUMERIC_FEATURES
        # column (unchanged default behavior); a list drops the rest. Persisted
        # with the artifact so inference uses the identical trained columns.
        self.features = features
        self.kw = kw
        self.clf = None
        self.calibrator: Optional[Calibrator] = None
        self.train_max: Optional[Tuple[int, int]] = None
        # Phase 7.2 ensemble (model="ensemble"): kw={"members": [...],
        # "combiner": "avg"|"meta"}. members/meta populated by _fit_ensemble.
        self.members: Dict[str, object] = {}
        self.meta = None
        self.meta_fold_spans: List[Tuple[int, int, int]] = []
        self.combiner: Optional[str] = None

    def _cols(self) -> List[str]:
        return feature_columns(self.features)

    def _member_clf(self, name: str):
        return MLRanker(name, seed=self.seed, features=self.features)._new_clf()

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
        mask = y.notna()
        f, yy = frame.loc[mask], y[mask].astype(int)
        self.train_max = (int(f["season"].max()),
                          int(f.query("season == season.max()")["week"].max()))
        if self.model_name == "ensemble":
            self._fit_ensemble(f, yy)
        else:
            cols = self._cols()
            X = f[cols]
            if self.model_name == "rf":
                X = X.fillna(-999.0)          # RF can't take NaN; sentinel is fine for trees
            self.clf = self._new_clf().fit(X, yy)
        if self.calibrate:
            self._fit_calibrator(f, yy)
        return self

    # -- Phase 7.2 ensemble (GBDT + RF) -------------------------------------- #
    def _fit_ensemble(self, frame: pd.DataFrame, y: pd.Series) -> None:
        """Fit each member on the full train pool (production quality), and --
        if ``combiner=="meta"`` -- a logistic meta-learner on strictly
        out-of-sample member predictions generated by expanding-season folds
        (the same discipline as the calibrator's ``cal_fold_spans``: no fold's
        meta-training row was ever predicted by a member that trained on it)."""
        members: List[str] = list(self.kw.get("members", ["gbdt", "rf"]))
        self.combiner = self.kw.get("combiner", "avg")
        cols = self._cols()
        self.members = {}
        for name in members:
            X = frame[cols]
            if name == "rf":
                X = X.fillna(-999.0)
            self.members[name] = self._member_clf(name).fit(X, y)
        if self.combiner == "meta":
            self._fit_meta(frame, y, cols, members)

    def _fit_meta(self, frame: pd.DataFrame, y: pd.Series, cols: List[str],
                 members: List[str]) -> None:
        seasons = sorted(frame["season"].unique().tolist())
        self.meta_fold_spans = []
        if len(seasons) < 2:
            self.meta = None
            return
        oos_p: Dict[str, List[np.ndarray]] = {m: [] for m in members}
        ys: List[np.ndarray] = []
        for s in seasons[1:]:
            tr = frame[frame["season"] < s]
            te = frame[frame["season"] == s]
            for name in members:
                Xtr, Xte = tr[cols], te[cols]
                if name == "rf":
                    Xtr, Xte = Xtr.fillna(-999.0), Xte.fillna(-999.0)
                fold = self._member_clf(name).fit(Xtr, y.loc[tr.index])
                oos_p[name].append(fold.predict_proba(Xte)[:, 1])
            ys.append(y.loc[te.index].to_numpy())
            self.meta_fold_spans.append(
                (int(s), int(tr["season"].min()), int(tr["season"].max())))
        from sklearn.linear_model import LogisticRegression
        X_meta = np.column_stack([_logit(np.concatenate(oos_p[m])) for m in members])
        self.meta = LogisticRegression(C=1e6, solver="lbfgs").fit(X_meta, np.concatenate(ys))

    def _oos_fold_predict(self, tr: pd.DataFrame, te: pd.DataFrame,
                         y: pd.Series) -> np.ndarray:
        """One expanding-fold OOS prediction, dispatching on model type --
        shared by the calibrator fit (any model) and, for ensembles, by the
        calibrator's ensemble folds. A single-model fold trains a fresh clf on
        ``tr`` and predicts ``te``; an ensemble fold trains a fresh temporary
        ensemble (members + its OWN nested meta-learner, if any, fit only on
        ``tr``'s own history) so no fold ever sees a prediction its members or
        meta-learner were trained on."""
        if self.model_name == "ensemble":
            temp = MLRanker("ensemble", seed=self.seed, features=self.features,
                            **self.kw)
            temp.fit(tr, y.loc[tr.index])
            return temp.predict_p_over(te, enforce=False)
        cols = self._cols()
        Xtr, Xte = tr[cols], te[cols]
        if self.model_name == "rf":
            Xtr, Xte = Xtr.fillna(-999.0), Xte.fillna(-999.0)
        fold = self._new_clf().fit(Xtr, y.loc[tr.index])
        return fold.predict_proba(Xte)[:, 1]

    def _fit_calibrator(self, frame: pd.DataFrame, y: pd.Series) -> None:
        """Fit the walk-forward calibrator on OUT-OF-SAMPLE base predictions.

        The base clf above is fit on ALL training rows (it produces the shipped
        numbers). The calibrator, however, must learn from predictions the base
        made on data it did NOT train on -- otherwise it corrects overconfident
        in-sample fits, not the real out-of-sample distortion. So we regenerate
        base predictions by EXPANDING-SEASON folds: for each season s (after the
        first), a fold model trains on seasons < s and predicts s. Those pooled
        OOS predictions -- none of which the fold ever trained on -- are what the
        calibrator is fit on. Fewer than two seasons of history -> no calibrator
        (identity), stated rather than faked."""
        spec = {"platt_permkt": ("platt", True),
                "platt_pooled": ("platt", False)}.get(self.calibrate)
        if spec is None:
            raise ValueError(f"unknown calibrate={self.calibrate!r} "
                             "(expected 'platt_permkt' or 'platt_pooled')")
        method, per_market = spec
        seasons = sorted(frame["season"].unique().tolist())
        if len(seasons) < 2:
            self.calibrator = None
            self.cal_fold_spans = []
            return
        ps, ys, ms = [], [], []
        # provenance + leakage witness: for each fold predicting season s, the
        # (min, max) train season -- a test asserts train_max < s (no fold ever
        # trains on the season it will help calibrate). See tests/test_leakage.py.
        self.cal_fold_spans: List[Tuple[int, int, int]] = []
        for s in seasons[1:]:
            tr = frame[frame["season"] < s]
            te = frame[frame["season"] == s]
            ps.append(self._oos_fold_predict(tr, te, y))
            ys.append(y.loc[te.index].to_numpy())
            ms.append(te["market"].to_numpy())
            self.cal_fold_spans.append(
                (int(s), int(tr["season"].min()), int(tr["season"].max())))
        self.calibrator = Calibrator(method, per_market).fit(
            np.concatenate(ps), np.concatenate(ys), np.concatenate(ms))

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

    def predict_p_over(self, frame: pd.DataFrame, enforce: bool = True,
                       raw: bool = False) -> np.ndarray:
        """Calibrated P(over) (the shipped quantity). ``raw=True`` returns the
        uncalibrated base probability -- used only by the audit; production and
        the pipeline call this with defaults and receive calibrated output, so
        the interface (p_over, and p_under = 1 - p_over downstream) is unchanged."""
        if enforce:
            self.assert_walk_forward(frame)
        if self.model_name == "ensemble":
            cols = self._cols()
            preds: Dict[str, np.ndarray] = {}
            for name, clf in self.members.items():
                X = frame[cols]
                if name == "rf":
                    X = X.fillna(-999.0)
                preds[name] = clf.predict_proba(X)[:, 1]
            if self.combiner == "meta" and self.meta is not None:
                order = list(self.members.keys())
                X_meta = np.column_stack([_logit(preds[m]) for m in order])
                p = self.meta.predict_proba(X_meta)[:, 1]
            else:
                p = np.mean(np.column_stack(list(preds.values())), axis=1)
        else:
            X = frame[self._cols()]
            if self.model_name == "rf":
                X = X.fillna(-999.0)
            p = self.clf.predict_proba(X)[:, 1]
        if self.calibrator is not None and not raw:
            p = self.calibrator.transform(p, frame["market"].to_numpy())
        return p

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str = MODEL_PATH_DEFAULT) -> str:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model_name": self.model_name, "seed": self.seed,
                     "calibrate": self.calibrate, "calibrator": self.calibrator,
                     "features": self.features,
                     "kw": self.kw, "clf": self.clf, "train_max": self.train_max,
                     "members": self.members, "meta": self.meta,
                     "meta_fold_spans": self.meta_fold_spans,
                     "combiner": self.combiner}, path)
        return path

    @classmethod
    def load(cls, path: str = MODEL_PATH_DEFAULT) -> "MLRanker":
        import joblib
        blob = joblib.load(path)
        obj = cls(blob["model_name"], blob["seed"],
                  calibrate=blob.get("calibrate"), features=blob.get("features"),
                  **blob.get("kw", {}))
        obj.clf, obj.train_max = blob["clf"], tuple(blob["train_max"])
        obj.calibrator = blob.get("calibrator")
        obj.members = blob.get("members", {})
        obj.meta = blob.get("meta")
        obj.meta_fold_spans = blob.get("meta_fold_spans", [])
        obj.combiner = blob.get("combiner")
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
