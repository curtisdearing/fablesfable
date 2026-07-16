"""Version-stable tabular content fingerprints.

Parquet is a storage format, not a content identity: writer/library versions
can change its bytes while preserving every cell.  These helpers hash a
canonical CSV stream instead.
"""

from __future__ import annotations

import csv
import hashlib
import math

import numpy as np
import pandas as pd


CANONICAL_CSV_VERSION = 1


class _DigestWriter:
    def __init__(self, digest: "hashlib._Hash") -> None:
        self.digest = digest

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8")
        self.digest.update(encoded)
        return len(value)


def _cell(value: object) -> str:
    """Encode one scalar with a type tag, so null, text, and numbers differ."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "null:"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        if math.isnan(value):
            return "null:"
        if math.isinf(value):
            return "float:+inf" if value > 0 else "float:-inf"
        return f"float:{format(0.0 if value == 0 else value, '.17g')}"
    if isinstance(value, (pd.Timestamp,)):
        return f"datetime:{value.isoformat()}"
    return f"str:{value}"


def canonical_csv_sha256(frame: pd.DataFrame, *, row_keys: list[str]) -> str:
    """Hash a deterministic, type-tagged UTF-8 CSV representation of ``frame``.

    Columns are lexicographic and rows are sorted by required unique business
    keys.  The hash intentionally identifies tabular content, not Parquet
    compression, metadata, dictionary layout, or writer version.
    """

    missing = sorted(set(row_keys) - set(frame.columns))
    if missing:
        raise ValueError(f"canonical hash row keys missing from frame: {missing}")
    if frame.duplicated(row_keys).any():
        raise ValueError("canonical hash row keys must uniquely identify rows")
    columns = sorted(frame.columns, key=str)
    if len({str(column) for column in columns}) != len(columns):
        raise ValueError("canonical hash requires unique string column names")

    work = frame.reset_index(drop=True)
    key_frame = pd.DataFrame({column: work[column].map(_cell) for column in row_keys})
    order = key_frame.sort_values(row_keys, kind="mergesort").index
    digest = hashlib.sha256()
    writer = csv.writer(_DigestWriter(digest), lineterminator="\n")
    writer.writerow([str(column) for column in columns])
    for row in work.loc[order, columns].itertuples(index=False, name=None):
        writer.writerow([_cell(value) for value in row])
    return digest.hexdigest()
