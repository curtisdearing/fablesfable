from types import SimpleNamespace

import pandas as pd
import pytest

from nflvalue.advanced_features import EXT_PBP_COLUMNS
from scripts import bootstrap_history as bootstrap


class FakeNFL:
    def __init__(self, missing_column=None):
        self.calls = []
        self.missing_column = missing_column

    def load_pbp(self, seasons, columns=None):
        season = seasons[0]
        self.calls.append(season)
        data = {column: [0] for column in EXT_PBP_COLUMNS}
        data["season"] = [season]
        data["season_type"] = ["REG"]
        if self.missing_column:
            data.pop(self.missing_column)
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame(data))

    def load_schedules(self):
        rows = [{"season": season, "week": 1, "game_type": "REG", "game_id": str(season)}
                for season in bootstrap.BASE_SEASONS]
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows))


def test_bootstrap_streams_one_season_at_a_time_and_validates(tmp_path):
    nfl = FakeNFL()
    pbp = tmp_path / "pbp.parquet"
    schedules = tmp_path / "schedules.parquet"
    rows = bootstrap.build_pbp(nfl, pbp)
    schedule_rows = bootstrap.build_schedules(nfl, schedules)
    assert nfl.calls == list(bootstrap.BASE_SEASONS)
    assert rows == len(bootstrap.BASE_SEASONS) == schedule_rows
    assert bootstrap.valid_parquet(pbp, EXT_PBP_COLUMNS, bootstrap.BASE_SEASONS)
    assert bootstrap.valid_parquet(schedules, bootstrap.SCHEDULE_REQUIRED, bootstrap.BASE_SEASONS)


def test_failed_rebuild_does_not_replace_existing_cache(tmp_path):
    destination = tmp_path / "pbp.parquet"
    destination.write_bytes(b"old-known-good")
    with pytest.raises(RuntimeError, match="lacks columns"):
        bootstrap.build_pbp(FakeNFL(missing_column="epa"), destination)
    assert destination.read_bytes() == b"old-known-good"
