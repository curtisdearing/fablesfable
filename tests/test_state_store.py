import io
import tarfile

import pytest

from scripts import state_store


def test_state_round_trip_is_checksummed_and_declared(tmp_path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "data" / "model.db").write_bytes(b"database-v1")
    (root / "data" / "latest.json").write_text('{"week": 3}')
    (root / "data" / "untracked.txt").write_text("do not archive")
    archive = tmp_path / "state.tar.gz"
    manifest = state_store.pack(archive, root)

    (root / "data" / "model.db").write_bytes(b"corrupt-local")
    restored = state_store.restore(archive, manifest["sha256"], root)
    assert (root / "data" / "model.db").read_bytes() == b"database-v1"
    assert set(restored) == {"data/latest.json", "data/model.db"}
    assert "data/untracked.txt" not in manifest["files"]


def test_restore_rejects_checksum_mismatch(tmp_path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "data" / "model.db").write_bytes(b"db")
    archive = tmp_path / "state.tar.gz"
    state_store.pack(archive, root)
    with pytest.raises(ValueError, match="checksum mismatch"):
        state_store.restore(archive, "0" * 64, root)


def test_restore_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    payload = b"owned"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../outside.db")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe member"):
        state_store.restore(archive, state_store.sha256(archive), tmp_path / "repo")
