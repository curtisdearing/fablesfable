"""Model-run rollback depth survives publication ingest.

The real prune/save shell of both workflows runs against a simulated `model-state` release (stub
`gh` backed by a JSON file; `--jq` evaluated by the real jq binary). Production has eight model
snapshots of rollback depth (state-*); publication ingest writes pubstate-* and must never count
or delete a state-* archive, nor the current pointer target."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"
STUB = r'''#!PYTHON
import json, subprocess, sys
db = "RELEASE_JSON"
rel = json.load(open(db))
a = sys.argv[1:]
if a[:2] == ["release", "view"]:
    obj = {"assets": [{"name": n, "createdAt": c} for n, c in rel["assets"]], "body": rel["body"]}
    field = a[a.index("--json") + 1]
    out = json.dumps({field: obj[field]})
    jq = a[a.index("--jq") + 1]
    print(subprocess.run(["jq", "-r", jq], input=out, capture_output=True, text=True, check=True).stdout, end="")
elif a[:2] == ["release", "delete-asset"]:
    rel["assets"] = [x for x in rel["assets"] if x[0] != a[3]]
    rel.setdefault("deleted", []).append(a[3])
    json.dump(rel, open(db, "w"))
else:
    sys.exit(f"unexpected gh {a}")
'''

pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="jq binary not available")


def _step(workflow, name):
    text = (WF / workflow).read_text()
    i = text.index(f"- name: {name}")
    body = text[i:]
    run = body[body.index("run: |") + len("run: |"):]
    lines = []
    for line in run.split("\n")[1:]:
        if line.strip() and not line.startswith(" " * 10):
            break
        lines.append(line[10:])
    return "\n".join(lines)


class Release:
    def __init__(self, tmp):
        self.path = tmp / "release.json"
        self.bin = tmp / "bin"
        self.bin.mkdir()
        (self.bin / "gh").write_text(STUB.replace("PYTHON", sys.executable).replace("RELEASE_JSON", str(self.path)))
        (self.bin / "gh").chmod(0o755)
        self.t = 0
        self.save({"assets": [], "body": "{}"})

    def load(self):
        return json.loads(self.path.read_text())

    def save(self, rel):
        self.path.write_text(json.dumps(rel))

    def upload(self, name):
        """What the save steps do: upload the archive, then move the pointer to it."""
        rel = self.load()
        self.t += 1
        rel["assets"].append([name, f"2026-09-{1 + self.t // 1440:02d}T{(self.t // 60) % 24:02d}:{self.t % 60:02d}:00Z"])
        rel["body"] = json.dumps({"schema_version": 1, "asset": name, "sha256": "0" * 64})
        self.save(rel)

    def prune(self, workflow, step):
        env = {"PATH": f"{self.bin}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "STATE_TAG": "model-state"}
        subprocess.run(["bash", "-c", "set -euo pipefail\n" + _step(workflow, step)], env=env, check=True,
                       capture_output=True, text=True)

    def names(self, prefix):
        return [n for n, _ in self.load()["assets"] if re.match(rf"^{prefix}-\d+-\d+\.tar\.gz$", n)]


def _model_run(rel, n):
    rel.upload(f"state-{n}-1.tar.gz")
    rel.prune("live-weekly.yml", "Keep the eight newest state archives")


def _ingest_run(rel, n):
    rel.upload(f"pubstate-{n}-1.tar.gz")
    before = set(rel.names("state"))
    rel.prune("publication-ingest.yml", "Keep the eight newest publication-ingest archives")
    assert set(rel.names("state")) == before                  # ingest never deletes a model snapshot


def test_ingest_never_erodes_model_rollback_depth(tmp_path):
    rel = Release(tmp_path)
    for n in range(1, 13):                            # 12 model runs, each followed by 3 publications
        _model_run(rel, 100 + n)
        for k in range(3):
            _ingest_run(rel, 1000 + 10 * n + k)
    states = rel.names("state")
    assert states == [f"state-{100 + n}-1.tar.gz" for n in range(5, 13)]      # the eight newest model snapshots
    assert len(rel.names("pubstate")) == 8
    current = json.loads(rel.load()["body"])["asset"]
    assert current in [n for n, _ in rel.load()["assets"]]
    deleted = rel.load()["deleted"]
    assert sorted(d for d in deleted if d.startswith("state-")) == [f"state-{100 + n}-1.tar.gz" for n in range(1, 5)]
    assert all(re.match(r"^(state|pubstate)-", d) for d in deleted)


def test_ingest_prune_keeps_the_current_pointer_even_when_it_is_old(tmp_path):
    rel = Release(tmp_path)
    for n in range(10):
        rel.upload(f"pubstate-{n}-1.tar.gz")
    rel2 = rel.load()
    rel2["body"] = json.dumps({"asset": "pubstate-0-1.tar.gz"})                # e.g. a manual rollback
    rel.save(rel2)
    rel.prune("publication-ingest.yml", "Keep the eight newest publication-ingest archives")
    names = rel.names("pubstate")
    assert "pubstate-0-1.tar.gz" in names and "pubstate-1-1.tar.gz" not in names and len(names) == 9


def test_model_prune_never_touches_pubstate_and_restore_accepts_both_names():
    live = (WF / "live-weekly.yml").read_text()
    ingest = (WF / "publication-ingest.yml").read_text()
    assert 'test("^state-[0-9]+-[0-9]+\\\\.tar\\\\.gz$")' in _step("live-weekly.yml", "Keep the eight newest state archives")
    assert 'test("^pubstate-[0-9]+-[0-9]+\\\\.tar\\\\.gz$")' in _step("publication-ingest.yml",
                                                                  "Keep the eight newest publication-ingest archives")
    assert 'asset="pubstate-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.tar.gz"' in ingest
    assert 'asset="state-${GITHUB_RUN_ID}' not in ingest
    for text in (live, ingest):
        assert "^(state|pubstate)-[0-9]+-[0-9]+\\.tar\\.gz$" in text
    pattern = re.compile(r"^(state|pubstate)-[0-9]+-[0-9]+\.tar\.gz$")
    assert pattern.match("pubstate-1-1.tar.gz") and pattern.match("state-1-1.tar.gz")
    assert not pattern.match("pubstate-1-1.tar.gz.evil") and not pattern.match("x-state-1-1.tar.gz")
