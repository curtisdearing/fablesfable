#!/usr/bin/env python3
"""Rebuild the historical/ caches the fablesfable frame build needs, offline-safe."""
import os, sys, urllib.request
import pandas as pd, pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")
os.makedirs(HIST, exist_ok=True)
sys.path.insert(0, ROOT)
from nflvalue.advanced_features import EXT_PBP_COLUMNS

SEASONS = [int(s) for s in os.environ.get("NFL_SEASONS", "2019,2020,2021,2022,2023,2024,2025").split(",")]
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.parquet"

def log(*a): print(*a, flush=True)

# 1. play-by-play
base_frames = []
for s in SEASONS:
    raw = os.path.join(ROOT, "data", "analysis_cache", f"pbp_raw_{s}.parquet")
    if not os.path.exists(raw):
        log(f"downloading pbp {s}...")
        urllib.request.urlretrieve(PBP_URL.format(s=s), raw)
    cols = [c for c in EXT_PBP_COLUMNS if c in pq.ParquetFile(raw).schema.names]
    df = pd.read_parquet(raw, columns=cols)
    for c in EXT_PBP_COLUMNS:
        if c not in df.columns: df[c] = pd.NA
    df = df[EXT_PBP_COLUMNS]
    if s <= 2023:
        base_frames.append(df)
    else:
        df.to_parquet(os.path.join(HIST, f"pbp_{s}.parquet"), index=False)
        log(f"pbp_{s}.parquet {len(df):,} rows")
    del df
pd.concat(base_frames, ignore_index=True).to_parquet(
    os.path.join(HIST, "historical_pbp.parquet"), index=False)
log(f"historical_pbp.parquet ({sum(len(f) for f in base_frames):,} rows)")
del base_frames

import nflreadpy as nfl

# 2. schedules/lines
sched = nfl.load_schedules().to_pandas()
sched[sched["season"].between(2019, 2023)].to_parquet(os.path.join(ROOT, "historical_lines.parquet"), index=False)
sched[sched["season"] >= 2024].to_parquet(os.path.join(HIST, "lines_extra.parquet"), index=False)
log("schedules done")

# 3. NGS receiving
try:
    n = nfl.load_nextgen_stats(stat_type="receiving").to_pandas()
except TypeError:
    n = nfl.load_nextgen_stats(seasons=True, stat_type="receiving").to_pandas()
n = n[n["season"].isin(SEASONS)]
n = n[n["week"] > 0]  # weekly rows only (week 0 = season aggregate)
n.to_parquet(os.path.join(HIST, "ngs_receiving.parquet"), index=False)
log(f"ngs_receiving {len(n):,} rows")

# 4. contracts
try:
    c = nfl.load_contracts().to_pandas()
    keep = [k for k in ["player", "position", "team", "year_signed", "years", "value", "apy", "is_active", "gsis_id"] if k in c.columns]
    c[keep].to_parquet(os.path.join(HIST, "contracts.parquet"), index=False)
    log(f"contracts {len(c):,} rows")
except Exception as e:
    log("contracts failed (non-fatal):", e)

# 5. FTN charting 2022+
from nflvalue import ftn_features
for s in [s for s in SEASONS if s >= ftn_features.FTN_SEASON_MIN]:
    try:
        log(f"ftn {s}: {ftn_features.refresh(s):,} plays")
    except Exception as e:
        log(f"ftn {s} failed (non-fatal):", e)

# 6. rosters / players meta / injuries (pre-warm the self-fetching caches)
from nflvalue.sources import rosters as rmod
r = rmod.fetch_rosters_weekly(SEASONS)
log(f"rosters_weekly {len(r):,} rows")
from nflvalue import context_features as ctx
log(f"players_meta {len(ctx.load_players_meta()):,}")
log(f"injuries {len(ctx.load_injury_history(SEASONS)):,}")
log("BOOTSTRAP COMPLETE")
