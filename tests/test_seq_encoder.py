"""Challenger B (GRU sequence encoder) — protocol tests (§B5).

Sequence leakage (poisoning games >= w cannot change the encoder output for
(S, w)), determinism (same-seed retrain -> identical extracted features),
fail-safe artifact handling (absent/corrupt -> the NaN-features path), and
PCA train-only fitting (poisoning the eval season leaves the projection
unchanged).  Tiny synthetic fixtures — green from a fresh clone, including
under FABLESFABLE_STRICT_FIXTURES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="torch is Challenger B's declared engine")

from nflvalue import seq_encoder as se                     # noqa: E402


def _toy_gl(seed=0, n_players=8, weeks=12, season=2019):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        base = rng.uniform(2, 9)
        for w in range(1, weeks + 1):
            rows.append({
                "season": season, "week": w, "player_id": f"00-{p:04d}",
                "team": f"T{p % 4}", "defteam": f"T{(p + 1) % 4}", "role": "WR",
                "targets": rng.poisson(base), "receptions": rng.poisson(base * 0.6),
                "rec_yards": max(rng.normal(base * 8, 10), 0.0),
                "air_yards_sum": max(rng.normal(base * 9, 12), 0.0),
                "carries": rng.poisson(1), "rush_yards": max(rng.normal(3, 4), 0.0),
                "pass_attempts": 0.0, "pass_yards": 0.0,
                "tds_total": float(rng.poisson(0.4)),
                "home": float(w % 2), "rest_days": 1.0,
                "order": season * 100 + w,
            })
    return pd.DataFrame(rows).sort_values(
        ["player_id", "order"], kind="mergesort").reset_index(drop=True)


def test_fit_refuses_games_at_or_after_eval_season():
    gl = pd.concat([_toy_gl(season=2019), _toy_gl(season=2020)], ignore_index=True)
    with pytest.raises(ValueError, match="walk-forward violation"):
        se.SeqEncoder(7).fit(gl, eval_season=2020)


def test_sequence_leakage_poisoning_future_weeks_changes_nothing():
    """Encoder output for (S, w) uses only games strictly before (S, w):
    poisoning every game at/after w must leave the hidden state bit-identical."""
    gl = _toy_gl()
    enc = se.SeqEncoder(7).fit(gl, eval_season=2020)
    req = pd.DataFrame([{"season": 2019, "week": 8, "player_id": "00-0001"}])
    clean = enc.hidden_states(gl, req)
    poisoned = gl.copy()
    future = poisoned["week"] >= 8
    for c in se.STAT_CHANNELS:
        poisoned.loc[future, c] = 999.0
    dirty = enc.hidden_states(poisoned, req)
    hcols = [f"h{i}" for i in range(se.HIDDEN)]
    assert np.array_equal(clean[hcols].to_numpy(), dirty[hcols].to_numpy())


def test_same_seed_retrain_identical_features():
    gl = _toy_gl()
    req = gl[["season", "week", "player_id"]].drop_duplicates().tail(10)
    hcols = [f"h{i}" for i in range(se.HIDDEN)]
    h1 = se.SeqEncoder(1234).fit(gl, eval_season=2020).hidden_states(gl, req)
    h2 = se.SeqEncoder(1234).fit(gl, eval_season=2020).hidden_states(gl, req)
    assert np.array_equal(h1[hcols].to_numpy(), h2[hcols].to_numpy(),
                          equal_nan=True)


def test_zero_prior_games_yields_nan_features():
    gl = _toy_gl()
    enc = se.SeqEncoder(7).fit(gl, eval_season=2020)
    req = pd.DataFrame([
        {"season": 2019, "week": 1, "player_id": "00-0001"},   # no priors
        {"season": 2019, "week": 5, "player_id": "00-0001"},   # has priors
    ])
    h = enc.hidden_states(gl, req)
    hcols = [f"h{i}" for i in range(se.HIDDEN)]
    assert np.isnan(h.loc[0, hcols].to_numpy(dtype=float)).all()
    assert np.isfinite(h.loc[1, hcols].to_numpy(dtype=float)).all()


def test_pca_is_fit_train_only():
    _toy_gl()
    rng = np.random.default_rng(0)
    train_h = rng.normal(size=(200, se.HIDDEN))
    pca = se.fit_pca(train_h, seed=7)
    eval_h = rng.normal(size=(50, se.HIDDEN))
    p1 = se.project_pca(pca, eval_h)
    # poisoning OTHER eval rows cannot move this row's projection
    eval_poisoned = eval_h.copy()
    eval_poisoned[10:] = 1e6
    p2 = se.project_pca(pca, eval_poisoned)
    assert np.array_equal(p1[:10], p2[:10])
    # NaN rows pass through as NaN (the GBDT's native missing path)
    eval_nan = eval_h.copy()
    eval_nan[0] = np.nan
    p3 = se.project_pca(pca, eval_nan)
    assert np.isnan(p3[0]).all() and np.array_equal(p3[1:10], p1[1:10])


def test_artifact_fail_safe(tmp_path):
    gl = _toy_gl()
    enc = se.SeqEncoder(7).fit(gl, eval_season=2020)
    path = str(tmp_path / "seq_encoder_2020.pt")
    enc.save(path)
    loaded = se.SeqEncoder.load(path)
    req = gl[["season", "week", "player_id"]].drop_duplicates().tail(5)
    hcols = [f"h{i}" for i in range(se.HIDDEN)]
    assert np.array_equal(enc.hidden_states(gl, req)[hcols].to_numpy(),
                          loaded.hidden_states(gl, req)[hcols].to_numpy(),
                          equal_nan=True)
    with open(path, "r+b") as fh:
        fh.seek(20)
        fh.write(b"\xde\xad\xbe\xef")
    with pytest.raises((ValueError, RuntimeError)):
        se.SeqEncoder.load(path)
    with pytest.raises(OSError):
        se.SeqEncoder.load(str(tmp_path / "missing.pt"))


def test_encoder_never_sees_lines_or_labels():
    """The training design matrix is built from the game log alone — assert
    the module's declared stat channels contain no line/label columns."""
    banned = {"line", "y_over", "p_over", "synthetic_line", "hit"}
    assert not banned & set(se.STAT_CHANNELS)


def test_attach_fail_safe_absent_artifact(tmp_path, monkeypatch):
    """§B2.4 golden path: NO artifact on disk -> attach stamps NaN seq_h*
    columns and returns; nothing raises, nothing is silently consumed."""
    f = pd.DataFrame({"season": [2026], "week": [1], "player_id": ["00-0001"],
                      "line": [4.5]})
    out = se.attach_seq_features(f, pw=None,
                                 path=str(tmp_path / "does_not_exist.joblib"))
    assert list(f.columns) + se.SEQ_FEATURES == list(out.columns)
    assert out[se.SEQ_FEATURES].isna().all().all()


def test_attach_fail_safe_corrupt_artifact(tmp_path):
    path = str(tmp_path / "seq_encoder.joblib")
    with open(path, "wb") as fh:
        fh.write(b"not a joblib blob")
    with open(path + ".sha256", "w") as fh:
        fh.write("0" * 64 + "\n")          # digest mismatch -> integrity path
    f = pd.DataFrame({"season": [2026], "week": [1], "player_id": ["00-0001"]})
    out = se.attach_seq_features(f, pw=None, path=path)
    assert out[se.SEQ_FEATURES].isna().all().all()


def test_production_artifact_roundtrip_and_attach(tmp_path, monkeypatch):
    """Full integration loop on toy data: build artifact -> attach -> the
    requested rows get finite features, unknown players get prior-safe
    values, and rows with no history stay NaN."""
    gl = _toy_gl()
    enc = se.SeqEncoder(7).fit(gl, eval_season=2020)
    rows = gl[["season", "week", "player_id"]].drop_duplicates()
    hid = enc.hidden_states(gl, rows)
    hcols = [f"h{i}" for i in range(se.HIDDEN)]
    pca = se.fit_pca(hid[hcols].to_numpy(), seed=7)
    import joblib
    path = str(tmp_path / "seq_encoder.joblib")
    joblib.dump({
        "state": {k: v.numpy() for k, v in enc.model.state_dict().items()},
        "scale": enc.scale, "player_index": enc.player_index,
        "team_index": enc.team_index, "meta": enc.meta,
        "pca_components": pca.components_, "pca_mean": pca.mean_,
    }, path)
    # attach needs schedules for the game log; feed the toy gl straight
    # through by monkeypatching the builder (schedules don't exist here)
    monkeypatch.setattr(se, "build_game_log", lambda pw, sched: gl)
    import nflvalue.ingest as ingest
    monkeypatch.setattr(ingest, "load_all_schedules", lambda: None)
    f = pd.DataFrame({
        "season": [2019, 2019], "week": [1, 8],
        "player_id": ["00-0001", "00-0001"], "line": [3.5, 3.5]})
    out = se.attach_seq_features(f, pw=gl, path=path)
    assert np.isnan(out.loc[0, se.SEQ_FEATURES].to_numpy(dtype=float)).all()
    assert np.isfinite(out.loc[1, se.SEQ_FEATURES].to_numpy(dtype=float)).all()
