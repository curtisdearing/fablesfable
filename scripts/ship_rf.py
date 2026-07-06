#!/usr/bin/env python3
"""One-off, checkpointed build of the Phase 7.2 production artifact: RF
(default hyperparams -- the HP search found no improvement worth shipping),
all 67 features (pruning was tested and NOT shipped -- see
docs/decisions_p7.md 7.2 -- it broke the passing_yards per-market guardrail),
platt_permkt calibration (7.1, unchanged).

RF's full-history fit (~25s) plus its 6 internal calibration folds (~10-20s
each) exceeds a single 45s call, and MLRanker.fit() runs them in one blocking
call -- so this script reproduces exactly what .fit() does, but in
checkpointed pieces across several invocations, then assembles the identical
object by hand and saves it.

Stages (run in order, each resumable):
  python3 scripts/ship_rf.py --stage main       # the shipped clf itself
  python3 scripts/ship_rf.py --stage cal_fold    # one calibration fold per call
  python3 scripts/ship_rf.py --stage assemble    # build + save the final MLRanker
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import config as cfgmod  # noqa: E402
from nflvalue import ml_ranker as mlr  # noqa: E402

FRAME_PATH = os.path.join(cfgmod.DATA_DIR, "ml_frame.parquet")
MAIN_CLF_PATH = os.path.join(cfgmod.DATA_DIR, "ship_rf_main.joblib")
CAL_FOLD_DIR = os.path.join(cfgmod.DATA_DIR, "ship_rf_cal_folds")
FINAL_PATH = "data/ml_ranker.joblib"


def run_main() -> None:
    if os.path.exists(MAIN_CLF_PATH):
        print("main clf already fitted")
        return
    frame = pd.read_parquet(FRAME_PATH)
    cols = mlr.feature_columns()
    y = frame["y_over"].astype(int)
    clf = mlr.MLRanker("rf")._new_clf().fit(frame[cols], y)
    train_max = (int(frame["season"].max()),
                int(frame.query("season == season.max()")["week"].max()))
    joblib.dump({"clf": clf, "train_max": train_max}, MAIN_CLF_PATH)
    print(f"main RF fitted on {len(frame):,} rows through {train_max}")


def run_cal_fold() -> None:
    os.makedirs(CAL_FOLD_DIR, exist_ok=True)
    frame = pd.read_parquet(FRAME_PATH)
    cols = mlr.feature_columns()
    seasons = sorted(frame["season"].unique().tolist())
    for s in seasons[1:]:
        out = os.path.join(CAL_FOLD_DIR, f"{s}.parquet")
        if os.path.exists(out):
            continue
        tr = frame[frame["season"] < s]
        te = frame[frame["season"] == s]
        clf = mlr.MLRanker("rf")._new_clf().fit(tr[cols], tr["y_over"].astype(int))
        p = clf.predict_proba(te[cols])[:, 1]
        pd.DataFrame({"season": s, "p": p, "y": te["y_over"].to_numpy(),
                     "market": te["market"].to_numpy(),
                     "train_min": int(tr["season"].min()),
                     "train_max": int(tr["season"].max())}).to_parquet(out, index=False)
        print(f"cal fold season {s}: train<{s} ({len(tr):,} rows) -> {len(te):,} preds")
        return
    print("all calibration folds done")


def run_assemble() -> None:
    from nflvalue.ml_ranker import Calibrator
    main = joblib.load(MAIN_CLF_PATH)
    frame = pd.read_parquet(FRAME_PATH)
    seasons = sorted(frame["season"].unique().tolist())
    fold_files = [os.path.join(CAL_FOLD_DIR, f"{s}.parquet") for s in seasons[1:]]
    missing = [f for f in fold_files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"missing calibration folds: {missing} -- run --stage cal_fold more")
    folds = [pd.read_parquet(f) for f in fold_files]
    pooled = pd.concat(folds, ignore_index=True)
    cal_fold_spans = [(int(f["season"].iloc[0]), int(f["train_min"].iloc[0]),
                      int(f["train_max"].iloc[0])) for f in folds]

    m = mlr.MLRanker("rf", calibrate="platt_permkt")
    m.clf = main["clf"]
    m.train_max = tuple(main["train_max"])
    m.calibrator = Calibrator("platt", True).fit(
        pooled["p"].to_numpy(), pooled["y"].to_numpy(), pooled["market"].to_numpy())
    m.cal_fold_spans = cal_fold_spans
    path = m.save(FINAL_PATH)
    print(f"assembled + saved production artifact -> {path}")
    print(f"train_max={m.train_max}, cal_fold_spans={cal_fold_spans}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "cal_fold", "assemble"], required=True)
    args = ap.parse_args()
    {"main": run_main, "cal_fold": run_cal_fold, "assemble": run_assemble}[args.stage]()


if __name__ == "__main__":
    main()
