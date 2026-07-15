#!/usr/bin/env python3
"""One-command refresh of the pattern batteries + player condition book.

Usage (repo root, weekly after update_results.py; add the new season as its
data lands):
    NFL_SEASONS=2019,2020,2021,2022,2023,2024,2025,2026 python3 refresh_condition_book.py

Steps (all walk-forward; mid-season runs only add completed weeks):
  1. analysis/bootstrap_data.py    pull missing nflverse caches
  2. build_week_inputs()           player-week table -> data/analysis_cache/
  3. analysis/pattern_battery.py   context + core battery
  4. analysis/extended_battery.py  cascades + obscure singles
  5. analysis/alldata_battery.py   max-n absence cascades (2019+)
  6. analysis/build_condition_book.py  per-player condition book -> book/

Outputs: book/player_condition_book.{parquet,csv}, book/stadium_splits.csv,
book/patterns*.json, book/refs.json. The live pipeline reads the book via
nflvalue/condition_book.py (context panel, display-only).
"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.makedirs("data/analysis_cache", exist_ok=True)
os.makedirs("book", exist_ok=True)

def run(script):
    print(f"\n=== {script} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join("analysis", script)], env=os.environ)
    if r.returncode != 0:
        sys.exit(f"{script} failed")

run("bootstrap_data.py")

print("\n=== rebuilding player-week inputs ===", flush=True)
sys.path.insert(0, ROOT)
from nflvalue.candidates import build_week_inputs
build_week_inputs().pw.to_parquet("data/analysis_cache/pw_cache.parquet")

for s in ["pattern_battery.py", "extended_battery.py", "alldata_battery.py",
          "build_condition_book.py"]:
    run(s)
print("\nDONE -> book/", flush=True)
