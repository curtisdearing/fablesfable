#!/usr/bin/env python3
"""Build the frozen, pregame-observable player-condition research frame."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from nflvalue import context_features, ingest
from nflvalue.features import build_player_week
from nflvalue.reproducibility import CANONICAL_CSV_VERSION, canonical_csv_sha256
from nflvalue.sources import rosters as roster_source


ROOT = Path(__file__).resolve().parents[1]
OUT_STATUSES = {"Out", "Doubtful"}
WEST_TEAMS = {"ARI", "DEN", "LAC", "LAR", "LV", "SEA", "SF"}
DB_POSITIONS = {"CB", "DB", "S", "FS", "SS"}
FRONT_POSITIONS = {"DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB", "EDGE"}
OL_POSITIONS = {"C", "G", "OG", "OT", "T", "OL", "LT", "LG", "RT", "RG"}
DEPTH_METRIC = {"QB": "pass_attempts", "RB": "carries", "WR": "targets", "TE": "targets"}
ACTIVE_STATUSES = {"ACT", "ACTIVE"}
RESERVE_STATUSES = {"RES", "IR", "PUP", "NFI", "SUS"}
MARKETS = {
    "QB": (("passing_yards", "pass_yards"), ("rushing_yards", "rush_yards")),
    "RB": (("rushing_yards", "rush_yards"), ("receiving_yards", "rec_yards"),
           ("receptions", "receptions"), ("anytime_td", "td_any")),
    "WR": (("receiving_yards", "rec_yards"), ("receptions", "receptions"),
           ("anytime_td", "td_any")),
    "TE": (("receiving_yards", "rec_yards"), ("receptions", "receptions"),
           ("anytime_td", "td_any")),
}


def _hash_paths(paths: list[Path]) -> dict[str, str | None]:
    hashes = {}
    for path in paths:
        if not path.exists():
            hashes[str(path.relative_to(ROOT))] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path.relative_to(ROOT))] = digest.hexdigest()
    return hashes


def team_schedule(schedules: pd.DataFrame) -> pd.DataFrame:
    """Flatten games to team rows without inferring anything from outcomes."""
    rows = []
    for game in schedules[schedules["game_type"] == "REG"].to_dict("records"):
        for side, opponent_side in (("home", "away"), ("away", "home")):
            row = {"season": int(game["season"]), "week": int(game["week"]),
                   "team": game.get(f"{side}_team"), "opponent": game.get(f"{opponent_side}_team"),
                   "is_home": int(side == "home")}
            for column in ("game_id", "gameday", "gametime", "weekday", "stadium", "roof",
                           "surface", "temp", "wind", "spread_line", "total_line", "div_game",
                           "referee", "overtime"):
                row[column] = game.get(column)
            spread = game.get("spread_line")
            row["team_spread"] = spread if side == "home" else (-spread if pd.notna(spread) else spread)
            rows.append(row)
    frame = pd.DataFrame(rows).dropna(subset=["team"])
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
    frame = frame.sort_values(["team", "gameday", "season", "week"])
    frame["rest_days"] = frame.groupby("team")["gameday"].diff().dt.days
    frame["after_overtime"] = frame.groupby("team")["overtime"].shift(1).eq(True)
    meetings = frame.groupby(["season", "team", "opponent"]).cumcount()
    frame["division_rematch"] = frame["div_game"].fillna(False).astype(bool) & (meetings > 0)
    return frame


def _history_index(pw: pd.DataFrame):
    histories = {}
    for role, metric in DEPTH_METRIC.items():
        subset = pw[pw["role"] == role].sort_values(["team", "player_id", "season", "week"])
        for (team, player), group in subset.groupby(["team", "player_id"], sort=False):
            keys = (group["season"].astype(int) * 100 + group["week"].astype(int)).tolist()
            histories[(team, player, role)] = (keys, group[metric].astype(float).tolist())
    return histories


def prior_depth(rosters: pd.DataFrame, pw: pd.DataFrame) -> pd.DataFrame:
    """Rank that week's roster using only each player's prior eight games."""
    histories = _history_index(pw)
    roster = rosters.rename(columns={"position": "role"}).copy()
    roster = roster[roster["role"].isin(DEPTH_METRIC)].copy()
    scored = []
    for row in roster.itertuples(index=False):
        key = int(row.season) * 100 + int(row.week)
        history_keys, values = histories.get((row.team, row.player_id, row.role), ([], []))
        cutoff = bisect.bisect_left(history_keys, key)
        prior = values[max(0, cutoff - 8):cutoff]
        scored.append(float(np.mean(prior)) if prior else 0.0)
    roster["prior_depth_score"] = scored
    roster = roster.sort_values(
        ["season", "week", "team", "role", "prior_depth_score", "player_id"],
        ascending=[True, True, True, True, False, True],
    )
    roster["depth_rank"] = roster.groupby(["season", "week", "team", "role"]).cumcount() + 1
    return roster[["season", "week", "team", "player_id", "role", "prior_depth_score", "depth_rank"]]


def official_absence_flags(depth: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    injuries = injuries.rename(columns={"gsis_id": "player_id"}).copy()
    out = injuries[injuries["report_status"].isin(OUT_STATUSES)].copy()
    ranked = out.merge(depth, on=["season", "week", "team", "player_id"], how="left")
    ranked["role"] = ranked["role"].fillna(ranked.get("position"))
    ranked = ranked[ranked["role"].isin(DEPTH_METRIC) & ranked["depth_rank"].notna()]
    flags = {}
    for row in ranked.itertuples(index=False):
        key = (int(row.season), int(row.week), row.team)
        flags.setdefault(key, {})[f"official_{row.role.lower()}{int(row.depth_rank)}_out"] = 1
    rows = []
    for key, values in flags.items():
        rows.append({"season": key[0], "week": key[1], "team": key[2], **values})
    return pd.DataFrame(rows)


def long_term_incumbent_vacancies(
    rosters: pd.DataFrame,
    pw: pd.DataFrame,
    *,
    min_prior_games: int = 3,
    min_absence_weeks: int = 2,
    candidate_horizon_weeks: int = 16,
) -> pd.DataFrame:
    """Find established players unavailable beyond a short-notice injury window.

    The existing depth rank deliberately considers only the current roster and
    is right for an official Out/Doubtful beneficiary study.  This companion
    cohort starts from *all* prior producers for the team/role, then asks
    whether the leading incumbent is unavailable in the current roster
    snapshot.  A player active for another team is treated as a transaction,
    never an injury vacancy.

    It is a research cohort, not a live model input: reserve/IR status is
    reliable enough to identify the gap, but historical roster snapshots do
    not by themselves timestamp a clean depth-chart replacement decision.
    """

    roster = rosters.rename(columns={"position": "role", "gsis_id": "player_id"}).copy()
    roster = roster[roster["role"].isin(DEPTH_METRIC)].copy()
    roster["player_id"] = roster["player_id"].fillna("").astype(str)
    roster = roster[roster["player_id"].ne("")].copy()
    status = roster["status"] if "status" in roster else pd.Series("", index=roster.index)
    roster["status"] = status.fillna("").astype(str).str.upper()
    roster = roster.sort_values(["season", "week", "team", "player_id", "status"]).drop_duplicates(
        ["season", "week", "team", "player_id"], keep="last"
    )
    histories = _history_index(pw)
    candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    for team, player_id, role in histories:
        candidates[(team, role)].add(player_id)
    active_elsewhere = {
        (int(row.season), int(row.week), row.player_id)
        for row in roster[roster["status"].isin(ACTIVE_STATUSES)].itertuples(index=False)
    }
    status_lookup = {
        (int(row.season), int(row.week), row.team, row.player_id): row.status
        for row in roster.itertuples(index=False)
    }
    rows = []
    for (season, team, role), weeks in roster.groupby(["season", "team", "role"], sort=True):
        team_weeks = sorted(int(value) for value in weeks["week"].unique())
        streaks: dict[str, int] = defaultdict(int)
        for index, week in enumerate(team_weeks):
            cutoff_key = int(season) * 100 + week
            earliest_key = int(season) * 100 + team_weeks[max(0, index - candidate_horizon_weeks)]
            ranked = []
            for player_id in candidates[(team, role)]:
                history_keys, values = histories[(team, player_id, role)]
                cutoff = bisect.bisect_left(history_keys, cutoff_key)
                if cutoff < min_prior_games or history_keys[cutoff - 1] < earliest_key:
                    continue
                prior = values[max(0, cutoff - 8):cutoff]
                if len(prior) >= min_prior_games:
                    ranked.append((float(np.mean(prior)), player_id))
            if not ranked:
                continue
            _, incumbent = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
            status = status_lookup.get((int(season), week, team, incumbent), "")
            active_here = status in ACTIVE_STATUSES
            active_other_team = (
                (int(season), week, incumbent) in active_elsewhere and not active_here
            )
            unavailable = not active_here and not active_other_team
            streaks[incumbent] = streaks[incumbent] + 1 if unavailable else 0
            long_term = unavailable and (
                status in RESERVE_STATUSES or streaks[incumbent] >= min_absence_weeks
            )
            if long_term:
                prefix = f"long_term_{role.lower()}1"
                rows.append({
                    "season": int(season), "week": week, "team": team,
                    f"{prefix}_unavailable": 1,
                    f"{prefix}_absence_weeks": streaks[incumbent],
                    f"{prefix}_reserve_status": int(status in RESERVE_STATUSES),
                })
    if not rows:
        return pd.DataFrame(columns=["season", "week", "team"])
    result = pd.DataFrame(rows)
    values = [column for column in result if column not in {"season", "week", "team"}]
    return result.groupby(["season", "week", "team"], as_index=False)[values].max()


def injury_counts(injuries: pd.DataFrame) -> pd.DataFrame:
    out = injuries[injuries["report_status"].isin(OUT_STATUSES)].copy()
    position = out["position"].fillna("").astype(str).str.upper()
    out["db_out"] = position.isin(DB_POSITIONS).astype(int)
    out["front_out"] = position.isin(FRONT_POSITIONS).astype(int)
    out["ol_out"] = position.isin(OL_POSITIONS).astype(int)
    return (out.groupby(["season", "week", "team"], as_index=False)
            .agg(db_outs=("db_out", "sum"), front_outs=("front_out", "sum"),
                 ol_outs=("ol_out", "sum")))


def _schedule_factors(frame: pd.DataFrame) -> pd.DataFrame:
    roof = frame["roof"].fillna("").astype(str).str.lower()
    surface = frame["surface"].fillna("").astype(str).str.lower()
    time_hour = pd.to_numeric(frame["gametime"].fillna("").astype(str).str.extract(r"^(\d+)")[0],
                              errors="coerce")
    frame["dome"] = roof.str.contains("dome|closed")
    frame["turf"] = surface.str.contains("turf|artificial")
    # Historical schedule weather is an observed proxy.  The name prevents it
    # from being mistaken for an archived pregame forecast.
    frame["observed_wind_15_plus_proxy"] = pd.to_numeric(frame["wind"], errors="coerce") >= 15
    frame["observed_cold_32_proxy"] = pd.to_numeric(frame["temp"], errors="coerce") <= 32
    frame["primetime"] = time_hour >= 19
    frame["short_rest"] = frame["rest_days"].between(0, 6)
    frame["post_bye"] = frame["rest_days"] >= 10
    frame["big_favorite"] = pd.to_numeric(frame["team_spread"], errors="coerce") <= -6.5
    frame["big_underdog"] = pd.to_numeric(frame["team_spread"], errors="coerce") >= 6.5
    frame["high_total"] = pd.to_numeric(frame["total_line"], errors="coerce") >= 48
    frame["division_game"] = frame["div_game"].fillna(False).astype(bool)
    frame["late_season"] = frame["week"] >= 14
    frame["body_clock_1pm_road"] = (~frame["is_home"].astype(bool) & frame["team"].isin(WEST_TEAMS)
                                        & (time_hour <= 13))
    return frame


def _player_prior_factors(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["player_id", "season", "week"]).copy()
    group = frame.groupby("player_id", sort=False)
    previous_carries = group["carries"].shift(1)
    previous_rec_yards = group["rec_yards"].shift(1)
    previous_rush_yards = group["rush_yards"].shift(1)
    previous_targets = group["targets"].shift(1)
    previous_roll_targets = group["roll_targets"].shift(1)
    frame["heavy_workload_last_game"] = previous_carries >= 22
    frame["receiving_100_last_game"] = previous_rec_yards >= 100
    frame["rushing_100_last_game"] = previous_rush_yards >= 100
    frame["target_spike_last_game"] = (previous_targets - previous_roll_targets) >= 5
    return frame


def _birthday(frame: pd.DataFrame, players: pd.DataFrame) -> pd.Series:
    meta = players.rename(columns={"gsis_id": "player_id"})[["player_id", "birth_date"]].drop_duplicates("player_id")
    merged = frame[["player_id", "gameday"]].merge(meta, on="player_id", how="left")
    game = pd.to_datetime(merged["gameday"], errors="coerce")
    birth = pd.to_datetime(merged["birth_date"], errors="coerce")
    # Circular day-of-year distance, with a 366-day denominator for Feb 29.
    distance = (game.dt.dayofyear - birth.dt.dayofyear).abs()
    return (np.minimum(distance, 366 - distance) <= 5).fillna(False)


def _revenge(frame: pd.DataFrame, rosters: pd.DataFrame) -> pd.Series:
    history = {}
    result = []
    roster = rosters.sort_values(["season", "week"])
    by_week = {(int(s), int(w)): group for (s, w), group in roster.groupby(["season", "week"])}
    for (season, week), indexes in frame.groupby(["season", "week"], sort=True).groups.items():
        for index in indexes:
            row = frame.loc[index]
            former = {team for team, count in history.get(row.player_id, {}).items()
                      if count >= 3 and team != row.team}
            result.append((index, row.opponent in former))
        for player in by_week.get((int(season), int(week)), pd.DataFrame()).itertuples(index=False):
            counts = history.setdefault(player.player_id, {})
            counts[player.team] = counts.get(player.team, 0) + 1
    return pd.Series(dict(result)).reindex(frame.index).fillna(False)


def to_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["td_any"] = frame["rush_tds"].fillna(0) + frame["rec_tds"].fillna(0)
    rows = []
    for role, markets in MARKETS.items():
        role_frame = frame[frame["role"] == role]
        for market, actual_column in markets:
            part = role_frame.copy()
            part["market"] = market
            part["actual"] = pd.to_numeric(part[actual_column], errors="coerce")
            rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    long = long.sort_values(["player_id", "market", "season", "week"])
    group = long.groupby(["player_id", "market"], sort=False)["actual"]
    long["prior_market_games"] = group.transform(lambda s: s.shift(1).rolling(8, min_periods=1).count())
    long["trailing_mean"] = group.transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    long["eligible"] = long["trailing_mean"].notna()
    long["over"] = (long["actual"] > long["trailing_mean"]).astype(int)
    touchdown = long["market"] == "anytime_td"
    long.loc[touchdown, "eligible"] = True
    long.loc[touchdown, "over"] = (long.loc[touchdown, "actual"] >= 1).astype(int)
    long["team_season"] = long["team"].astype(str) + "-" + long["season"].astype(str)
    return long


def build_frame(pbp: pd.DataFrame, schedules: pd.DataFrame, rosters: pd.DataFrame,
                injuries: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    pw = build_player_week(pbp, rosters=rosters)
    schedule = team_schedule(schedules)
    frame = pw.merge(schedule, on=["season", "week", "team"], how="inner")
    depth = prior_depth(rosters, pw)
    frame = frame.merge(depth[["season", "week", "team", "player_id", "depth_rank"]],
                        on=["season", "week", "team", "player_id"], how="left")
    absences = official_absence_flags(depth, injuries)
    if not absences.empty:
        frame = frame.merge(absences, on=["season", "week", "team"], how="left")
    long_term = long_term_incumbent_vacancies(rosters, pw)
    if not long_term.empty:
        frame = frame.merge(long_term, on=["season", "week", "team"], how="left")
    counts = injury_counts(injuries)
    frame = frame.merge(counts.rename(columns={"db_outs": "own_db_outs",
                                                "front_outs": "own_front_outs"}),
                        on=["season", "week", "team"], how="left")
    frame = frame.merge(
        counts[["season", "week", "team", "db_outs", "front_outs"]].rename(
            columns={"team": "opponent", "db_outs": "opponent_db_outs",
                     "front_outs": "opponent_front_outs"}),
        on=["season", "week", "opponent"], how="left",
    )
    for column in ("own_db_outs", "own_front_outs", "ol_outs", "opponent_db_outs",
                   "opponent_front_outs"):
        if column not in frame:
            frame[column] = 0
        frame[column] = frame[column].fillna(0).astype(int)
    frame["opponent_db_2_plus"] = frame["opponent_db_outs"] >= 2
    frame["opponent_front_2_plus"] = frame["opponent_front_outs"] >= 2
    frame["own_ol_2_plus"] = frame["ol_outs"] >= 2
    for role in ("qb", "rb", "wr", "te"):
        for rank in (1, 2, 3):
            column = f"official_{role}{rank}_out"
            if column not in frame:
                frame[column] = 0
            frame[column] = frame[column].fillna(0).astype(int)
        for suffix in ("unavailable", "absence_weeks", "reserve_status"):
            column = f"long_term_{role}1_{suffix}"
            if column not in frame:
                frame[column] = 0
            frame[column] = frame[column].fillna(0).astype(int)
    frame = _schedule_factors(frame)
    frame = _player_prior_factors(frame)
    frame["birthday_window_5"] = _birthday(frame, players).to_numpy()
    frame["revenge"] = _revenge(frame, rosters)
    frame["is_depth1"] = frame["depth_rank"] == 1
    frame["is_depth2"] = frame["depth_rank"] == 2
    return to_market_frame(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "factor_frame.parquet")
    parser.add_argument("--refresh-context", action="store_true")
    args = parser.parse_args()
    pbp = ingest.load_all_pbp()
    schedules = ingest.load_all_schedules()
    seasons = sorted(int(value) for value in pbp["season"].unique())
    rosters = roster_source.fetch_rosters_weekly(seasons, force_refresh=args.refresh_context)
    injuries = context_features.load_injury_history(seasons, refresh=args.refresh_context)
    players = context_features.load_players_meta(refresh=args.refresh_context)
    frame = build_frame(pbp, schedules, rosters, injuries, players)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    sources = [ROOT / "historical" / "historical_pbp.parquet", ROOT / "historical_lines.parquet",
               ROOT / "historical" / "rosters_weekly.parquet", ROOT / "historical" / "injuries.parquet",
               ROOT / "historical" / "players_meta.parquet"]
    metadata = {"schema_version": 2, "rows": len(frame), "seasons": seasons,
                "source_sha256": _hash_paths(sources),
                "frame_canonical_csv_sha256": canonical_csv_sha256(
                    frame, row_keys=["season", "week", "game_id", "team", "player_id", "market"]
                ),
                "canonical_csv_version": CANONICAL_CSV_VERSION,
                "weather_warning": "temp/wind are historical observed proxies, not archived forecast snapshots"}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
