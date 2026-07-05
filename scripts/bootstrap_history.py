"""One-shot rebuild of every gitignored historical/ cache from free nflverse
sources. The caches are regenerable by design (.gitignore: *.parquet); this
script IS the regeneration path, replacing the nfl_data_py-era
historical/download_history.py.

    python3 scripts/bootstrap_history.py            # everything missing
    python3 scripts/bootstrap_history.py --force    # re-pull everything

Column policy (Phase 6 decision, docs/decisions_p6.md): the 2019-2023 base
pbp parquet is rebuilt with EXT_PBP_COLUMNS + P6_EXTRA below -- NOT all ~397
nflverse columns -- so the whole history fits a 4GB box. Widening later =
add the column here and re-run. Per-season files (2024+) match.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HIST = os.path.join(ROOT, "historical")

from nflvalue.advanced_features import FULL_PBP_COLUMNS  # noqa: E402

# Single source of truth for kept pbp columns lives in advanced_features
# (FULL_PBP_COLUMNS = base + extended + Phase-6 tiers); ingest.refresh keeps
# the same set for new seasons.
PBP_KEEP = FULL_PBP_COLUMNS

BASE_SEASONS = [2019, 2020, 2021, 2022, 2023]
EXTRA_SEASONS = [2024, 2025]          # 2026 hasn't kicked off (July 2026)
ALL_SEASONS = BASE_SEASONS + EXTRA_SEASONS


def log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def _pull_pbp_season(nfl, season: int) -> pd.DataFrame:
    df = nfl.load_pbp(seasons=[season]).to_pandas()
    have = [c for c in PBP_KEEP if c in df.columns]
    missing = [c for c in PBP_KEEP if c not in df.columns]
    if missing:
        log(f"pbp {season}: nflverse lacks {missing} -- filling NaN")
    out = df[have].copy()
    for c in missing:
        out[c] = pd.NA
    del df
    gc.collect()
    return out


def build_base_pbp(nfl, force: bool) -> None:
    path = os.path.join(HIST, "historical_pbp.parquet")
    if os.path.exists(path) and not force:
        log("base pbp exists, skipping")
        return
    parts = []
    for s in BASE_SEASONS:
        log(f"pbp {s} ...")
        parts.append(_pull_pbp_season(nfl, s))
        log(f"pbp {s}: {len(parts[-1]):,} rows")
    base = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    base.to_parquet(path, index=False)
    log(f"base pbp written: {len(base):,} rows x {len(base.columns)} cols")
    del base
    gc.collect()


def build_extra_pbp(nfl, force: bool) -> None:
    for s in EXTRA_SEASONS:
        path = os.path.join(HIST, f"pbp_{s}.parquet")
        if os.path.exists(path) and not force:
            log(f"pbp_{s} exists, skipping")
            continue
        log(f"pbp {s} ...")
        df = _pull_pbp_season(nfl, s)
        df.to_parquet(path, index=False)
        log(f"pbp_{s} written: {len(df):,} rows")
        del df
        gc.collect()


def build_schedules(nfl, force: bool) -> None:
    base = os.path.join(ROOT, "historical_lines.parquet")     # ingest.BASE_LINES
    extra = os.path.join(HIST, "lines_extra.parquet")
    if os.path.exists(base) and os.path.exists(extra) and not force:
        log("schedules exist, skipping")
        return
    sched = nfl.load_schedules().to_pandas()
    sched[sched["season"].isin(BASE_SEASONS)].to_parquet(base, index=False)
    from nflvalue.ingest import SCHED_COLS
    ex = sched[sched["season"] >= 2024]
    keep = [c for c in SCHED_COLS if c in ex.columns]
    ex[keep].to_parquet(extra, index=False)
    log(f"schedules written: base {sched['season'].isin(BASE_SEASONS).sum():,} rows"
        f" (all cols), extra {len(ex):,} rows ({len(keep)} cols)")


def build_rosters(force: bool) -> None:
    from nflvalue.sources import rosters as rostersmod
    if force and os.path.exists(rostersmod.CACHE_PATH):
        os.remove(rostersmod.CACHE_PATH)
    r = rostersmod.fetch_rosters_weekly(ALL_SEASONS)
    log(f"rosters_weekly: {len(r):,} rows, seasons {sorted(r['season'].unique())}")


def build_context(force: bool) -> None:
    from nflvalue import context_features as cf
    if force:
        for p in (cf.PLAYERS_META, cf.INJURIES):
            if os.path.exists(p):
                os.remove(p)
    meta = cf.load_players_meta(refresh=not os.path.exists(cf.PLAYERS_META))
    log(f"players_meta: {len(meta):,} rows")
    inj = cf.load_injury_history(ALL_SEASONS)
    log(f"injuries: {len(inj):,} rows, seasons {sorted(inj['season'].unique()) if len(inj) else []}")


def build_ngs(nfl, force: bool) -> None:
    for st in ("receiving", "passing"):
        path = os.path.join(HIST, f"ngs_{st}.parquet")
        if os.path.exists(path) and not force:
            log(f"ngs_{st} exists, skipping")
            continue
        n = nfl.load_nextgen_stats(stat_type=st).to_pandas()   # all seasons (2016+)
        n = n[n["week"] > 0]
        n.to_parquet(path, index=False)
        log(f"ngs_{st}: {len(n):,} rows, {n['season'].min()}-{n['season'].max()}")


def build_contracts(nfl, force: bool) -> None:
    path = os.path.join(HIST, "contracts.parquet")
    if os.path.exists(path) and not force:
        log("contracts exist, skipping")
        return
    con = nfl.load_contracts().to_pandas()
    con[["gsis_id", "player", "position", "year_signed", "years",
         "value", "apy", "is_active"]].to_parquet(path, index=False)
    log(f"contracts: {len(con):,} rows")


def build_ftn(force: bool) -> None:
    from nflvalue import ftn_features
    import nflreadpy as nfl
    for s in [s for s in ALL_SEASONS if s >= ftn_features.FTN_SEASON_MIN]:
        path = ftn_features.cache_path(s)
        if os.path.exists(path) and not force:
            log(f"ftn_{s} exists, skipping")
            continue
        # feasibility record (6.1): log the FULL free-subset schema once so the
        # man/zone answer is on the record, then cache the usual columns.
        if s == 2022:
            raw = nfl.load_ftn_charting(seasons=[s]).to_pandas()
            log(f"FTN free-subset columns ({s}): {sorted(raw.columns)}")
            del raw
            gc.collect()
        n = ftn_features.refresh(s)
        log(f"ftn_{s}: {n:,} rows")


def build_snap_counts(nfl, force: bool) -> None:
    path = os.path.join(HIST, "snap_counts.parquet")
    if os.path.exists(path) and not force:
        log("snap_counts exist, skipping")
        return
    sc = nfl.load_snap_counts(seasons=ALL_SEASONS).to_pandas()
    keep = [c for c in ("game_id", "season", "week", "player", "pfr_player_id",
                        "position", "team", "offense_snaps", "offense_pct") if c in sc.columns]
    sc[keep].to_parquet(path, index=False)
    log(f"snap_counts: {len(sc):,} rows")


def build_trades(nfl, force: bool) -> None:
    path = os.path.join(HIST, "trades.parquet")
    if os.path.exists(path) and not force:
        log("trades exist, skipping")
        return
    if not hasattr(nfl, "load_trades"):
        log("nflreadpy lacks load_trades -- revenge-by-trade will need a fallback")
        return
    tr = nfl.load_trades().to_pandas()
    tr.to_parquet(path, index=False)
    log(f"trades: {len(tr):,} rows ({sorted(tr.columns)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(HIST, exist_ok=True)
    import nflreadpy as nfl

    steps = [
        ("base_pbp", lambda: build_base_pbp(nfl, args.force)),
        ("extra_pbp", lambda: build_extra_pbp(nfl, args.force)),
        ("schedules", lambda: build_schedules(nfl, args.force)),
        ("rosters", lambda: build_rosters(args.force)),
        ("context", lambda: build_context(args.force)),
        ("ngs", lambda: build_ngs(nfl, args.force)),
        ("contracts", lambda: build_contracts(nfl, args.force)),
        ("ftn", lambda: build_ftn(args.force)),
        ("snap_counts", lambda: build_snap_counts(nfl, args.force)),
        ("trades", lambda: build_trades(nfl, args.force)),
    ]
    failed = []
    for name, fn in steps:
        try:
            fn()
        except Exception:
            failed.append(name)
            log(f"STEP FAILED: {name}\n{traceback.format_exc()}")
    log(f"DONE. failed steps: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
