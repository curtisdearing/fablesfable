"""Time-travel audit: can historical QB/OL availability be replayed as of kickoff?

Read-only.  Answers two questions that gate any numerical personnel test:
  1. Does the cached injury history carry a per-row publication clock?
  2. Is nflverse ``home_qb_id/away_qb_id`` (the input to the ranker's
     ``qb_continuity``) a pregame projection or the realized starter?  For
     completed games we compare it with the passer who threw the team's first
     pass attempt (pbp), which is the realized starter.

    python analysis/personnel_matchup_asof_audit.py INJURIES_PARQUET GAMES_CSV PBP_PARQUET [...]
"""

import json
import sys

import pandas as pd


def main(inj_path, games_path, *pbp_paths):
    inj = pd.read_parquet(inj_path)
    clock_cols = [c for c in inj.columns if any(k in c.lower() for k in
                                                 ("date", "time", "modified", "published"))]
    games = pd.read_csv(games_path, dtype=str)
    res = {"injury_cache": {"rows": int(len(inj)), "columns": list(inj.columns),
                            "publication_clock_columns": clock_cols}}
    unplayed = games[(games["season"] == "2026") & games["result"].isna()]
    res["games_csv_2026_unplayed"] = {
        "games": int(len(unplayed)),
        "with_qb_id": int((unplayed["home_qb_id"].notna() | unplayed["away_qb_id"].notna()).sum())}
    per = {}
    for p in pbp_paths:
        pbp = pd.read_parquet(p, columns=["game_id", "season", "posteam", "play_id",
                                          "pass_attempt", "passer_player_id", "season_type"])
        pbp = pbp[(pbp["season_type"] == "REG") & (pbp["pass_attempt"] == 1)
                  & pbp["passer_player_id"].notna()].sort_values(["game_id", "play_id"])
        first = pbp.groupby(["game_id", "posteam"])["passer_player_id"].first()
        g = games[games["game_id"].isin(pbp["game_id"].unique())]
        n = agree = 0
        for r in g.itertuples(index=False):
            for team, qb in ((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id)):
                act = first.get((r.game_id, team))
                if act is None or pd.isna(qb):
                    continue
                n += 1
                agree += int(act == qb)
        season = int(pbp["season"].iloc[0])
        per[season] = {"team_games": n, "schedule_qb_equals_first_passer": agree,
                       "rate": round(agree / n, 4) if n else None}
    res["schedule_qb_vs_realized_first_passer"] = per
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
