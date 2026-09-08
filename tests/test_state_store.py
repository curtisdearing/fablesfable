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


def _archive_with(tmp_path, name: str, payload: bytes = b"x"):
    archive = tmp_path / "member.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return archive


def test_weekly_drop_is_production_state(tmp_path):
    """The Pages build publishes the drop that the payload names only when the
    document is on disk; only the Wednesday runner ever generated it. Carrying
    it in state is what keeps a deploy-on-push or a T-90 run from replacing
    the week's report with the no-current-report notice."""
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "drops").mkdir()
    (root / "data" / "weekly_props.json").write_text('{"season": 2026, "week": 1}')
    (root / "drops" / "props_week_2026_1.html").write_text("<h1>2026 Week 1</h1>")
    (root / "drops" / "props_week_2026_1_t90.html").write_text("<h1>2026 Week 1 t90</h1>")
    (root / "drops" / "notes.md").write_text("not state")
    archive = tmp_path / "state.tar.gz"
    manifest = state_store.pack(archive, root)
    assert set(manifest["files"]) == {
        "data/weekly_props.json",
        "drops/props_week_2026_1.html",
        "drops/props_week_2026_1_t90.html",
    }

    # A cold runner: the payload restores but the document must come with it.
    cold = tmp_path / "cold"
    restored = state_store.restore(archive, manifest["sha256"], cold)
    assert "drops/props_week_2026_1.html" in restored
    assert (cold / "drops" / "props_week_2026_1.html").read_text() == "<h1>2026 Week 1</h1>"


def test_restore_rejects_roots_and_names_the_declaration_does_not_cover(tmp_path):
    # A root that no declared glob starts with is unsafe, not merely undeclared:
    # the archive gets no say in where it may write.
    archive = _archive_with(tmp_path, "reports/latest.html")
    with pytest.raises(ValueError, match="unsafe member"):
        state_store.restore(archive, state_store.sha256(archive), tmp_path / "r1")
    # Inside an allowed root, only declared names restore.
    archive = _archive_with(tmp_path, "drops/HOW-TO-PUSH.md")
    with pytest.raises(ValueError, match="undeclared state"):
        state_store.restore(archive, state_store.sha256(archive), tmp_path / "r2")
    assert state_store.STATE_ROOTS == {"data", "drops"}
