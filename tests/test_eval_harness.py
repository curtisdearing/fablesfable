"""Accuracy harness: registry schema, gates, drift check contract."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, "analysis", "eval_harness.py")


def run(*args):
    return subprocess.run([sys.executable, HARNESS, *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_registry_written_with_schema(tmp_path):
    out = f"data/accuracy_registry_test.json"
    r = run("--output", out)
    assert r.returncode == 0, r.stderr
    reg = json.load(open(os.path.join(ROOT, out)))
    for key in ["schema_version", "generated", "git_head", "holdout_policy",
                "accept_gates", "inputs", "metrics"]:
        assert key in reg, key
    assert reg["accept_gates"]["ranker_log_loss"] < 0
    assert isinstance(reg["inputs"], dict) and reg["inputs"]
    os.remove(os.path.join(ROOT, out))


def test_check_mode_detects_no_drift(tmp_path):
    out = "data/accuracy_registry_test2.json"
    assert run("--output", out).returncode == 0
    r = run("--check", "--output", out)
    assert r.returncode == 0 and "unchanged" in r.stdout
    os.remove(os.path.join(ROOT, out))


def test_check_mode_fails_without_registry():
    r = run("--check", "--output", "data/definitely_missing_registry.json")
    assert r.returncode == 1
