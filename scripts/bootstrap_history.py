#!/usr/bin/env python3
"""Build and validate the frozen historical caches used by scheduled runs.

Play-by-play is fetched one season at a time, projected to the columns the
application actually consumes, and streamed to one parquet file.  Both output
files are written beside their destination and atomically replaced only after
schema and season validation succeeds.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue.advanced_features import EXT_PBP_COLUMNS  # noqa: E402


HIST = ROOT / "historical"
PBP_PATH = HIST / "historical_pbp.parquet"
LINES_PATH = ROOT / "historical_lines.parquet"
BASE_SEASONS = tuple(range(2019, 2024))
SCHEDULE_REQUIRED = {"season", "week", "game_type", "game_id"}


def _to_pandas(frame):
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


def _seasons_in(path: Path) -> set[int]:
    table = pq.read_table(path, columns=["season"])
    return {int(v) for v in table.column("season").to_pylist() if v is not None}


def valid_parquet(path: Path, required_columns: Iterable[str], seasons: Iterable[int]) -> bool:
    """Return whether *path* is readable and has the expected schema/cohort."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        names = set(pq.ParquetFile(path).schema_arrow.names)
        return set(required_columns).issubset(names) and _seasons_in(path) == set(seasons)
    except (OSError, ValueError, pa.ArrowException):
        return False


def _load_pbp_season(nfl, season: int):
    """Use projection pushdown when supported, while tolerating older clients."""
    try:
        frame = nfl.load_pbp(seasons=[season], columns=EXT_PBP_COLUMNS)
    except TypeError:
        frame = nfl.load_pbp(seasons=[season])
    frame = _to_pandas(frame)
    missing = sorted(set(EXT_PBP_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(f"play-by-play season {season} lacks columns: {missing}")
    frame = frame.loc[frame["season"].astype(int) == season, EXT_PBP_COLUMNS]
    if frame.empty:
        raise RuntimeError(f"play-by-play season {season} returned no rows")
    return frame


def build_pbp(nfl, destination: Path = PBP_PATH) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    writer = None
    schema = None
    rows = 0
    try:
        for season in BASE_SEASONS:
            frame = _load_pbp_season(nfl, season)
            table = pa.Table.from_pandas(frame, preserve_index=False, schema=schema, safe=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="snappy")
            writer.write_table(table)
            rows += len(frame)
        writer.close()
        writer = None
        if not valid_parquet(tmp, EXT_PBP_COLUMNS, BASE_SEASONS):
            raise RuntimeError("new play-by-play cache failed validation")
        os.replace(tmp, destination)
        return rows
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)


def build_schedules(nfl, destination: Path = LINES_PATH) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        schedules = _to_pandas(nfl.load_schedules())
        missing = sorted(SCHEDULE_REQUIRED - set(schedules.columns))
        if missing:
            raise RuntimeError(f"schedules lack columns: {missing}")
        schedules = schedules[schedules["season"].isin(BASE_SEASONS)].copy()
        if set(schedules["season"].astype(int)) != set(BASE_SEASONS):
            raise RuntimeError("schedules did not contain every requested base season")
        schedules.to_parquet(tmp, index=False)
        if not valid_parquet(tmp, SCHEDULE_REQUIRED, BASE_SEASONS):
            raise RuntimeError("new schedule cache failed validation")
        os.replace(tmp, destination)
        return len(schedules)
    finally:
        tmp.unlink(missing_ok=True)


def bootstrap(*, force: bool = False, nfl=None) -> dict[str, object]:
    if nfl is None:
        import nflreadpy as nfl

    pbp_valid = valid_parquet(PBP_PATH, EXT_PBP_COLUMNS, BASE_SEASONS)
    schedules_valid = valid_parquet(LINES_PATH, SCHEDULE_REQUIRED, BASE_SEASONS)
    result: dict[str, object] = {"pbp": "reused", "schedules": "reused"}
    if force or not pbp_valid:
        result["pbp"] = build_pbp(nfl)
    if force or not schedules_valid:
        result["schedules"] = build_schedules(nfl)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = bootstrap(force=args.force)
    print(f"[bootstrap] play-by-play={result['pbp']}; schedules={result['schedules']}")


if __name__ == "__main__":
    main()
