"""Phase 7.3 -- schema migrations and model-artifact integrity.

The live SQLite database is a DURABLE RELEASE ASSET: the weekly automation
downloads it from a GitHub release, appends a week, and re-uploads it. Two
consequences drive this file:

* ``CREATE TABLE IF NOT EXISTS`` is a no-op against a table that already
  exists, so without versioned migrations a newly added column would never
  appear on the deployed DB, and the code would read NULL-that-isn't-there
  from real production rows.
* The model artifact travels the same path, and a corrupted or truncated
  artifact still *loads* -- it just scores differently. That has to be fatal,
  not silent.

Offline, deterministic, temp-dir only.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from nflvalue import db as dbmod
from nflvalue import ml_ranker as mlr


# --------------------------------------------------------------------------- #
# Schema versioning
# --------------------------------------------------------------------------- #
def test_fresh_database_is_stamped_at_current_version(tmp_path):
    conn = dbmod.connect(str(tmp_path / "fresh.db"))
    assert dbmod.user_version(conn) == dbmod.SCHEMA_VERSION
    conn.close()


def test_migrations_are_idempotent(tmp_path):
    """Reconnecting must not re-run migrations or change the stamp -- the
    weekly job opens this DB every run."""
    path = str(tmp_path / "twice.db")
    conn = dbmod.connect(path)
    first = dbmod.user_version(conn)
    conn.close()

    conn = dbmod.connect(path)
    assert dbmod.user_version(conn) == first
    assert dbmod.migrate(conn) == first
    conn.close()


def test_pre_versioning_database_is_stamped_without_losing_rows(tmp_path):
    """The real upgrade path: a production DB written before Phase 7.3
    reports user_version=0 but is full of history. It must be adopted, not
    rebuilt, and every existing row must survive byte-for-byte."""
    path = str(tmp_path / "legacy.db")

    # Build a "legacy" DB exactly as the old connect() would have: tables, no
    # version stamp, then real rows.
    raw = sqlite3.connect(path)
    for ddl in dbmod.SCHEMA.values():
        raw.execute(ddl)
    raw.execute("INSERT INTO player_week (season, week, player_id, player_name, "
                "team, role, targets, receptions, rec_yards) "
                "VALUES (2023, 10, '00-LEGACY', 'Legacy Player', 'KC', 'WR', 9, 6, 88.5)")
    raw.commit()
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    raw.close()

    conn = dbmod.connect(path)
    assert dbmod.user_version(conn) == dbmod.SCHEMA_VERSION

    rows = conn.execute("SELECT season, week, player_id, player_name, targets, "
                        "receptions, rec_yards FROM player_week").fetchall()
    assert rows == [(2023, 10, "00-LEGACY", "Legacy Player", 9.0, 6.0, 88.5)], (
        "migration altered or dropped pre-existing production rows")
    conn.close()


def test_additive_column_does_not_orphan_existing_rows(tmp_path):
    """Prove the additive contract with a real ALTER: rows written before the
    column existed keep their values and read NULL (not a fabricated default)
    for the new field."""
    path = str(tmp_path / "additive.db")
    conn = dbmod.connect(path)
    dbmod.upsert(conn, "player_week", [{
        "season": 2023, "week": 10, "player_id": "00-OLD",
        "player_name": "Old Row", "team": "KC", "role": "WR", "rec_yards": 88.5,
    }], ["season", "week", "player_id"])

    added = dbmod.add_column_if_missing(conn, "player_week",
                                        "phase7_probe", "REAL")
    assert added is True
    assert dbmod.add_column_if_missing(conn, "player_week",
                                       "phase7_probe", "REAL") is False, \
        "add_column_if_missing must be idempotent"

    row = conn.execute("SELECT rec_yards, phase7_probe FROM player_week "
                       "WHERE player_id='00-OLD'").fetchone()
    assert row[0] == 88.5, "pre-existing value was disturbed by the migration"
    assert row[1] is None, "new column invented a value for a historical row"
    conn.close()


def test_refuses_to_open_a_database_from_a_newer_release(tmp_path):
    """A DB written by a LATER version of the code must not be silently
    downgraded -- that would drop columns the newer release is writing. Fail
    closed and loudly."""
    path = str(tmp_path / "future.db")
    conn = dbmod.connect(path)
    conn.execute(f"PRAGMA user_version={dbmod.SCHEMA_VERSION + 5}")
    conn.commit()

    with pytest.raises(RuntimeError, match="NEWER"):
        dbmod.migrate(conn)
    conn.close()


def test_declared_version_matches_the_migration_table():
    """Guard against the classic mistake: adding a migration and forgetting to
    bump SCHEMA_VERSION (or vice versa)."""
    assert dbmod.SCHEMA_VERSION == max(dbmod.MIGRATIONS), (
        "SCHEMA_VERSION and MIGRATIONS disagree")
    assert set(dbmod.MIGRATIONS) == set(range(1, dbmod.SCHEMA_VERSION + 1)), (
        "migration versions must be contiguous from 1")


# --------------------------------------------------------------------------- #
# Model artifact integrity
# --------------------------------------------------------------------------- #
def _tiny_fitted_ranker():
    """Smallest honest fit: a few rows of the real feature contract."""
    cols = mlr.feature_columns()
    n = 40
    frame = pd.DataFrame({c: [0.1 * (i % 7) for i in range(n)] for c in cols})
    frame["season"] = [2021 + (i % 3) for i in range(n)]
    frame["week"] = [1 + (i % 17) for i in range(n)]
    y = pd.Series([i % 2 for i in range(n)])
    return mlr.MLRanker(model="gbdt").fit(frame, y), frame


def test_save_writes_a_digest_and_load_accepts_it(tmp_path):
    model, _ = _tiny_fitted_ranker()
    path = str(tmp_path / "ranker.joblib")
    model.save(path)

    assert os.path.exists(path + ".sha256")
    with open(path + ".sha256", encoding="utf-8") as fh:
        recorded = fh.read().strip()
    assert recorded == mlr.artifact_sha256(path)

    reloaded = mlr.MLRanker.load(path)          # must not raise
    assert reloaded.model_name == model.model_name
    assert reloaded.train_max == model.train_max


def test_load_refuses_a_tampered_artifact(tmp_path):
    """The core of 7.3: a corrupted artifact still deserialises fine and would
    happily emit numbers. Scoring must be refused instead."""
    model, _ = _tiny_fitted_ranker()
    path = str(tmp_path / "ranker.joblib")
    model.save(path)

    with open(path, "r+b") as fh:          # flip one byte in the payload
        fh.seek(len(open(path, 'rb').read()) // 2)
        b = fh.read(1)
        fh.seek(-1, os.SEEK_CUR)
        fh.write(bytes([b[0] ^ 0xFF]))

    with pytest.raises(mlr.ArtifactIntegrityError, match="integrity"):
        mlr.MLRanker.load(path)


def test_load_refuses_a_truncated_artifact(tmp_path):
    """Partial download / interrupted write -- the realistic release-asset
    failure, not a malicious one."""
    model, _ = _tiny_fitted_ranker()
    path = str(tmp_path / "ranker.joblib")
    model.save(path)

    size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        fh.truncate(size // 2)

    with pytest.raises(mlr.ArtifactIntegrityError):
        mlr.MLRanker.load(path)


def test_missing_sidecar_is_tolerated_for_legacy_artifacts(tmp_path):
    """Artifacts fitted before 7.3 have no digest. Absence is not evidence of
    corruption, so it must not block scoring -- only DISAGREEMENT is fatal."""
    model, _ = _tiny_fitted_ranker()
    path = str(tmp_path / "ranker.joblib")
    model.save(path)
    os.remove(path + ".sha256")

    reloaded = mlr.MLRanker.load(path)
    assert reloaded.model_name == model.model_name


def test_digest_is_stable_across_repeated_hashing(tmp_path):
    model, _ = _tiny_fitted_ranker()
    path = str(tmp_path / "ranker.joblib")
    model.save(path)
    assert mlr.artifact_sha256(path) == mlr.artifact_sha256(path)
