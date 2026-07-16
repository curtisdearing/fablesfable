"""Feature-subset knob (config ml_ranker.features): selection, artifact
self-containment, and legacy default behavior."""
import numpy as np
import pandas as pd
import pytest

import nflvalue.ml_ranker as mlr


FULL = mlr.NUMERIC_FEATURES + [f"mkt_{m}" for m in mlr.MARKETS7] + [f"pos_{p}" for p in mlr.POSITIONS]
LEAN = ["p_over", "z", "mean", "sd", "line", "mkt_receptions", "pos_WR"]


@pytest.fixture()
def subset_cfg(monkeypatch):
    def fake_load():
        return {"ml_ranker": {"features": list(LEAN)}}
    monkeypatch.setattr("nflvalue.config.load_config", fake_load)
    yield


@pytest.fixture()
def frame():
    rng = np.random.default_rng(0)
    n = 400
    f = pd.DataFrame({c: rng.normal(size=n) for c in FULL})
    f["season"] = np.where(np.arange(n) < 300, 2021, 2022)
    f["week"] = 1 + (np.arange(n) % 17)
    return f, pd.Series((rng.random(n) > 0.5).astype(float))


def test_absent_key_means_full_set(monkeypatch):
    monkeypatch.setattr("nflvalue.config.load_config", lambda: {"ml_ranker": {}})
    assert mlr.feature_columns() == FULL
    monkeypatch.setattr("nflvalue.config.load_config",
                        lambda: {"ml_ranker": {"features": None}})
    assert mlr.feature_columns() == FULL


def test_subset_honored_and_order_preserved(subset_cfg):
    cols = mlr.feature_columns()
    assert set(cols) == set(LEAN)
    assert cols == [c for c in FULL if c in set(LEAN)]  # full-set order


def test_unknown_column_rejected(monkeypatch):
    monkeypatch.setattr("nflvalue.config.load_config",
                        lambda: {"ml_ranker": {"features": ["p_over", "nope_col"]}})
    with pytest.raises(ValueError):
        mlr.feature_columns()


def test_artifact_carries_features(subset_cfg, frame, tmp_path, monkeypatch):
    f, y = frame
    tr = f[f["season"] == 2021]
    model = mlr.MLRanker(model="gbdt", max_iter=10).fit(tr, y[tr.index])
    assert model.features == mlr.feature_columns()
    path = model.save(str(tmp_path / "m.joblib"))
    # config flips back to FULL after training -> loaded artifact must still
    # predict with its own recorded subset, not the live config
    monkeypatch.setattr("nflvalue.config.load_config", lambda: {"ml_ranker": {}})
    loaded = mlr.MLRanker.load(path)
    assert loaded.features == model.features
    te = f[f["season"] == 2022]
    p = loaded.predict_p_over(te)
    assert p.shape == (len(te),) and np.isfinite(p).all()
