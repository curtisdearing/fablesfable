"""Advanced process metrics: strategic aggression, NGS efficiency-vs-luck,
red-zone roles, O-line health, QB continuity, contract year, weather.

The user's spec, implemented walk-forward (every rolling value is
shift(1)-then-aggregate, the same leak-proof idiom features.py uses):

STRATEGIC AGGRESSION (coaching intent, not results)
  team_neutral_proe   mean nflverse ``pass_oe`` over NEUTRAL plays only:
                      1st/2nd down, Q1-Q3, score within +/-7, win prob
                      20-80% -- strips garbage-time and desperation noise.
  team_edp            early-down passing: pass rate on 1st & 10 (leading
                      indicator of a high-ceiling offense).
  team_pace           median seconds between snaps on neutral plays within
                      the same drive (possession-maximizing offenses).
  team_epa_play       rolling EPA/play -- is the aggression EFFICIENT or
                      just variance-chasing?
  The PROE-vs-total edge the user described is left to the GBDT: it sees
  both ``team_neutral_proe`` and ``total_line`` and learns the interaction
  (trees split on exactly such conjunctions).

NGS RECEIVING / PASSING (2016+, tracking-based; efficiency vs luck)
  ngs_separation      avg separation at target -- open-ness is skill+scheme.
  ngs_ay_share        percent share of team INTENDED air yards -- the
                      ceiling predictor.
  ngs_yac_aoe         YAC above expected -- actual >> expected flags
                      unsustainable luck (regression candidate).
  team_cpoe           rolling completion-%-over-expected of the player's
                      offense (pbp ``cpoe``) -- separates receiver skill
                      from elite-QB gravy.

ROLES / PERSONNEL
  rz_tgt_share,       red-zone (yardline_100 <= 20) target/carry share --
  rz_carry_share      THE anytime-TD driver raw volume misses.
  team_shotgun_rate,  formation tendencies (shotgun, no-huddle) -- proxy
  team_no_huddle_rate for personnel/formation intent until FTN charting
                      (2022+) earns a column.
  qb_continuity       share of the team's trailing pass attempts thrown by
                      the QB the schedule PROJECTS to start this week --
                      a WR's rolling stats mean less under a new arm.
  oline_outs          own-team offensive linemen listed Out/Doubtful.
  is_contract_year    final season of the player's latest known deal
                      (nflverse/OTC) -- the structured slice of "more TDs,
                      more money". Specific bonus clauses aren't structured
                      free data; they flow through news -> context_ledger.
  age_years           from DOB at gameday.

WEATHER
  temp / wind from the schedules table (domes/closed roofs neutralized to
  70F / 0mph). Honesty note: historical values are OBSERVED, live values
  come from forecast -- close but not identical distributions; recorded in
  decisions_p3-5.md.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import PBP_COLUMNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")

EXT_ONLY = ["play_id", "down", "ydstogo", "yardline_100", "score_differential", "qtr", "wp",
            "xpass", "pass_oe", "cpoe", "shotgun", "no_huddle",
            "game_seconds_remaining", "sack", "qb_hit", "pass", "rush", "fixed_drive"]
EXT_PBP_COLUMNS = list(dict.fromkeys(PBP_COLUMNS + EXT_ONLY))

# Phase 6: every pbp column any tier consumes, in ONE list -- ingest keeps
# exactly these for new seasons and scripts/bootstrap_history.py rebuilds the
# base with them, so a column needed by weather fitting (6.4) or O-line/
# scramble effects (6.5) can't silently vanish from future caches.
P6_ONLY = ["goal_to_go", "qb_scramble", "qb_dropback",
           "field_goal_attempt", "field_goal_result", "kick_distance",
           "kicker_player_id", "kicker_player_name", "extra_point_result",
           "weather", "roof", "surface", "stadium", "home_team", "away_team",
           "drive", "touchdown", "td_player_id"]
FULL_PBP_COLUMNS = list(dict.fromkeys(EXT_PBP_COLUMNS + P6_ONLY))

OL_POS = {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"}
FEATURES = [
    "team_neutral_proe", "team_edp", "team_pace", "team_epa_play",
    "team_shotgun_rate", "team_no_huddle_rate", "team_cpoe",
    "ngs_separation", "ngs_ay_share", "ngs_yac_aoe",
    "rz_tgt_share", "rz_carry_share",
    "qb_continuity", "oline_outs", "is_contract_year", "age_years",
    "temp", "wind",
]


def load_pbp_ext() -> pd.DataFrame:
    """Base 2019-2023 (all 397 cols on disk) + per-season extended files."""
    frames = [pd.read_parquet(os.path.join(HIST, "historical_pbp.parquet"),
                              columns=EXT_PBP_COLUMNS)]
    for fn in sorted(os.listdir(HIST)):
        if fn.startswith("pbp_") and fn.endswith(".parquet"):
            f = pd.read_parquet(os.path.join(HIST, fn))
            missing = [c for c in EXT_PBP_COLUMNS if c not in f.columns]
            if missing:
                raise RuntimeError(f"{fn} lacks extended columns {missing[:4]} -- "
                                   "re-run ingest.refresh(force=True)")
            frames.append(f[EXT_PBP_COLUMNS])
    df = pd.concat(frames, ignore_index=True)
    return df[df["season_type"] == "REG"].reset_index(drop=True)


def _roll(g: pd.core.groupby.SeriesGroupBy, span: int = 6) -> pd.Series:
    return g.transform(lambda s: s.shift(1).ewm(span=span, min_periods=1).mean())


# --------------------------------------------------------------------------- #
# Team tendencies (strategic aggression + formations + CPOE)
# --------------------------------------------------------------------------- #
def build_team_tendencies(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp.copy()
    is_play = (p["pass"] == 1) | (p["rush"] == 1)
    neutral = (is_play & p["down"].isin([1, 2]) & (p["qtr"] <= 3)
               & (p["score_differential"].abs() <= 7)
               & p["wp"].between(0.20, 0.80))
    early = is_play & (p["down"] == 1) & (p["ydstogo"] == 10)

    # seconds/play within (game, posteam, drive), neutral situations only
    p["_sec"] = p["game_seconds_remaining"]
    p = p.sort_values(["game_id", "fixed_drive", "_sec"],
                      ascending=[True, True, False]).reset_index(drop=True)
    same_drive = ((p["game_id"] == p["game_id"].shift())
                  & (p["fixed_drive"] == p["fixed_drive"].shift())
                  & (p["posteam"] == p["posteam"].shift()))
    gap = (p["_sec"].shift() - p["_sec"]).where(same_drive)
    p["_snap_gap"] = gap.where((gap > 0) & (gap <= 45))

    def agg(mask: pd.Series, col: str, how: str = "mean") -> pd.DataFrame:
        d = p[mask].groupby(["season", "week", "posteam"])[col]
        return (d.median() if how == "median" else d.mean()).rename(col).reset_index()

    out = agg(neutral & p["pass_oe"].notna(), "pass_oe")
    out = out.merge(p[early].groupby(["season", "week", "posteam"])["pass"]
                    .mean().rename("edp").reset_index(), how="outer",
                    on=["season", "week", "posteam"])
    out = out.merge(agg(neutral & p["_snap_gap"].notna(), "_snap_gap", "median")
                    .rename(columns={"_snap_gap": "pace"}), how="outer",
                    on=["season", "week", "posteam"])
    out = out.merge(agg(is_play, "epa"), how="outer", on=["season", "week", "posteam"])
    out = out.merge(agg(is_play, "shotgun"), how="outer", on=["season", "week", "posteam"])
    out = out.merge(agg(is_play, "no_huddle"), how="outer", on=["season", "week", "posteam"])
    out = out.merge(agg((p["pass"] == 1) & p["cpoe"].notna(), "cpoe"), how="outer",
                    on=["season", "week", "posteam"])

    out = out.sort_values(["posteam", "season", "week"]).reset_index(drop=True)
    g = out.groupby("posteam")
    for src, dst in (("pass_oe", "team_neutral_proe"), ("edp", "team_edp"),
                     ("pace", "team_pace"), ("epa", "team_epa_play"),
                     ("shotgun", "team_shotgun_rate"), ("no_huddle", "team_no_huddle_rate"),
                     ("cpoe", "team_cpoe")):
        out[dst] = _roll(g[src])
    keep = ["season", "week", "posteam", "team_neutral_proe", "team_edp", "team_pace",
            "team_epa_play", "team_shotgun_rate", "team_no_huddle_rate", "team_cpoe"]
    return out[keep].rename(columns={"posteam": "team"})


# --------------------------------------------------------------------------- #
# Player red-zone shares
# --------------------------------------------------------------------------- #
def build_player_redzone(pbp: pd.DataFrame) -> pd.DataFrame:
    """Rolling red-zone shares WITHOUT shift: the roll at a player's row
    (s, w) INCLUDES week w, because consumers look these up AS-OF STRICTLY
    BEFORE the candidate week (see AsOfLookup). Rolled-with-shift exact-week
    joins caused a missingness leak (a row only existed for weeks the player
    saw the red zone -- NaN itself revealed this week's usage; caught at
    +21pts of impossible OOS accuracy and fixed here)."""
    rz = pbp[pbp["yardline_100"] <= 20]
    tgt = (rz[rz["pass"] == 1].dropna(subset=["receiver_player_id"])
           .groupby(["season", "week", "posteam", "receiver_player_id"])
           .size().rename("rz_tgt").reset_index()
           .rename(columns={"receiver_player_id": "player_id"}))
    car = (rz[rz["rush"] == 1].dropna(subset=["rusher_player_id"])
           .groupby(["season", "week", "posteam", "rusher_player_id"])
           .size().rename("rz_car").reset_index()
           .rename(columns={"rusher_player_id": "player_id"}))
    team_t = rz[rz["pass"] == 1].groupby(["season", "week", "posteam"]).size().rename("team_rz_tgt")
    team_c = rz[rz["rush"] == 1].groupby(["season", "week", "posteam"]).size().rename("team_rz_car")

    df = tgt.merge(car, how="outer", on=["season", "week", "posteam", "player_id"])
    df = df.merge(team_t.reset_index(), how="left", on=["season", "week", "posteam"])
    df = df.merge(team_c.reset_index(), how="left", on=["season", "week", "posteam"])
    for c in ("rz_tgt", "rz_car", "team_rz_tgt", "team_rz_car"):
        df[c] = df[c].fillna(0.0)
    df["_tgt_share"] = df["rz_tgt"] / df["team_rz_tgt"].replace(0, np.nan)
    df["_car_share"] = df["rz_car"] / df["team_rz_car"].replace(0, np.nan)
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = df.groupby("player_id")
    df["rz_tgt_share"] = g["_tgt_share"].transform(
        lambda s: s.rolling(16, min_periods=1).mean())
    df["rz_carry_share"] = g["_car_share"].transform(
        lambda s: s.rolling(16, min_periods=1).mean())
    return df[["season", "week", "player_id", "rz_tgt_share", "rz_carry_share"]]


# --------------------------------------------------------------------------- #
# NGS (walk-forward rolling of weekly tracking metrics)
# --------------------------------------------------------------------------- #
def build_ngs_receiving(path: Optional[str] = None) -> pd.DataFrame:
    path = path or os.path.join(HIST, "ngs_receiving.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["season", "week", "player_id",
                                     "ngs_separation", "ngs_ay_share", "ngs_yac_aoe"])
    n = pd.read_parquet(path)
    n = (n.rename(columns={"player_gsis_id": "player_id"})
         .dropna(subset=["player_id"])
         .sort_values(["player_id", "season", "week"]).reset_index(drop=True))
    # NO shift here: NGS rows exist only for qualifying weeks, so an
    # exact-week join would leak this week's volume via missingness (see
    # build_player_redzone note). AsOfLookup reads strictly-prior rows.
    g = n.groupby("player_id")
    n["ngs_separation"] = g["avg_separation"].transform(
        lambda s: s.ewm(span=4, min_periods=1).mean())
    n["ngs_ay_share"] = g["percent_share_of_intended_air_yards"].transform(
        lambda s: s.ewm(span=4, min_periods=1).mean())
    n["ngs_yac_aoe"] = g["avg_yac_above_expectation"].transform(
        lambda s: s.ewm(span=4, min_periods=1).mean())
    return n[["season", "week", "player_id", "ngs_separation", "ngs_ay_share", "ngs_yac_aoe"]]


# --------------------------------------------------------------------------- #
# QB continuity: trailing attempts share of THIS week's projected starter
# --------------------------------------------------------------------------- #
def build_qb_continuity(pbp: pd.DataFrame, schedules: pd.DataFrame) -> Dict[Tuple, float]:
    att = (pbp[pbp["pass_attempt"] == 1].dropna(subset=["passer_player_id"])
           .groupby(["season", "week", "posteam", "passer_player_id"])
           .size().rename("att").reset_index())
    att = att.sort_values(["posteam", "season", "week"])
    out: Dict[Tuple, float] = {}
    sched = schedules[schedules["game_type"] == "REG"]
    hist_by_team: Dict[str, List] = {}
    for r in att.itertuples(index=False):
        hist_by_team.setdefault(r.posteam, []).append((r.season, r.week, r.passer_player_id, r.att))
    for g in sched.itertuples(index=False):
        for team, qb in ((g.home_team, getattr(g, "home_qb_id", None)),
                         (g.away_team, getattr(g, "away_qb_id", None))):
            if not qb or not isinstance(qb, str):
                continue
            rows = [x for x in hist_by_team.get(team, [])
                    if (x[0], x[1]) < (g.season, g.week)][-60:]
            total = sum(x[3] for x in rows)
            mine = sum(x[3] for x in rows if x[2] == qb)
            if total > 0:
                out[(int(g.season), int(g.week), team)] = round(mine / total, 4)
    return out


# --------------------------------------------------------------------------- #
# Contract year (walk-forward: deals signed on/before the season)
# --------------------------------------------------------------------------- #
def contract_year_lookup(path: Optional[str] = None) -> Dict[Tuple[str, int], int]:
    path = path or os.path.join(HIST, "contracts.parquet")
    if not os.path.exists(path):
        return {}
    con = pd.read_parquet(path).dropna(subset=["gsis_id", "year_signed", "years"])
    out: Dict[Tuple[str, int], int] = {}
    for season in range(2019, 2033):
        known = con[con["year_signed"] <= season]
        latest = known.sort_values("year_signed").groupby("gsis_id").tail(1)
        for r in latest.itertuples(index=False):
            end = int(r.year_signed) + int(r.years) - 1
            out[(r.gsis_id, season)] = int(season == end)
    return out


class AsOfLookup:
    """Per-player 'latest value STRICTLY BEFORE (season, week)' lookup.

    The anti-missingness-leak primitive: every candidate row gets the
    player's most recent prior value regardless of whether the player has a
    row at the candidate week itself, so NaN means 'no prior history' --
    never 'nothing happened THIS week'."""

    def __init__(self, df: pd.DataFrame, value_cols: List[str]):
        self.value_cols = value_cols
        self.data: Dict[str, Tuple[List[int], List[Tuple]]] = {}
        d = df.sort_values(["player_id", "season", "week"])
        for pid, grp in d.groupby("player_id"):
            wkeys = (grp["season"].astype(int) * 100 + grp["week"].astype(int)).tolist()
            vals = list(grp[value_cols].itertuples(index=False, name=None))
            self.data[pid] = (wkeys, vals)

    def get(self, player_id: str, season: int, week: int) -> Tuple:
        entry = self.data.get(player_id)
        nan_row = tuple(np.nan for _ in self.value_cols)
        if entry is None:
            return nan_row
        wkeys, vals = entry
        import bisect
        i = bisect.bisect_left(wkeys, season * 100 + week)  # strictly before
        return vals[i - 1] if i > 0 else nan_row


# --------------------------------------------------------------------------- #
# The pack
# --------------------------------------------------------------------------- #
class AdvancedPack:
    def __init__(self, pbp: Optional[pd.DataFrame] = None,
                 schedules: Optional[pd.DataFrame] = None,
                 injuries: Optional[pd.DataFrame] = None,
                 players_meta: Optional[pd.DataFrame] = None):
        from .context_features import (OUT_STATUSES, load_injury_history,
                                       load_players_meta)
        pbp = pbp if pbp is not None else load_pbp_ext()
        if schedules is None:
            from .ingest import load_all_schedules
            schedules = load_all_schedules()

        tt = build_team_tendencies(pbp)
        self.team: Dict[Tuple, Dict] = {
            (int(r.season), int(r.week), r.team): r._asdict()
            for r in tt.itertuples(index=False)}
        self.rz = AsOfLookup(build_player_redzone(pbp),
                             ["rz_tgt_share", "rz_carry_share"])
        self.ngs = AsOfLookup(build_ngs_receiving(),
                              ["ngs_separation", "ngs_ay_share", "ngs_yac_aoe"])
        self.qbc = build_qb_continuity(pbp, schedules)
        self.contract = contract_year_lookup()

        seasons = sorted({int(s) for s in pbp["season"].unique()})
        inj = injuries if injuries is not None else load_injury_history(seasons)
        self.ol_out: Dict[Tuple, int] = {}
        if len(inj):
            d = inj[inj["report_status"].isin(OUT_STATUSES) & inj["position"].isin(OL_POS)]
            for (s, w, t), grp in d.groupby(["season", "week", "team"]):
                self.ol_out[(int(s), int(w), str(t))] = int(len(grp))

        meta = players_meta if players_meta is not None else load_players_meta()
        self.dob = {r.player_id: pd.Timestamp(r.birth_date)
                    for r in meta.itertuples(index=False) if pd.notna(r.birth_date)}
        self.weather: Dict[str, Tuple[float, float]] = {}
        for g in schedules.itertuples(index=False):
            roof = str(getattr(g, "roof", "") or "").lower()
            if roof in ("dome", "closed"):
                self.weather[g.game_id] = (70.0, 0.0)
            else:
                t = getattr(g, "temp", None)
                w = getattr(g, "wind", None)
                self.weather[g.game_id] = (
                    float(t) if t is not None and pd.notna(t) else np.nan,
                    float(w) if w is not None and pd.notna(w) else np.nan)

    def attach(self, cands: pd.DataFrame) -> pd.DataFrame:
        cands = cands.copy()
        rows = {f: [] for f in FEATURES}
        for r in cands.itertuples(index=False):
            key = (int(r.season), int(r.week))
            t = self.team.get((*key, r.team), {})
            rows["team_neutral_proe"].append(t.get("team_neutral_proe", np.nan))
            rows["team_edp"].append(t.get("team_edp", np.nan))
            rows["team_pace"].append(t.get("team_pace", np.nan))
            rows["team_epa_play"].append(t.get("team_epa_play", np.nan))
            rows["team_shotgun_rate"].append(t.get("team_shotgun_rate", np.nan))
            rows["team_no_huddle_rate"].append(t.get("team_no_huddle_rate", np.nan))
            rows["team_cpoe"].append(t.get("team_cpoe", np.nan))
            rz = self.rz.get(r.player_id, *key)
            rows["rz_tgt_share"].append(rz[0])
            rows["rz_carry_share"].append(rz[1])
            ngs = self.ngs.get(r.player_id, *key)
            rows["ngs_separation"].append(ngs[0])
            rows["ngs_ay_share"].append(ngs[1])
            rows["ngs_yac_aoe"].append(ngs[2])
            rows["qb_continuity"].append(self.qbc.get((*key, r.team), np.nan))
            rows["oline_outs"].append(self.ol_out.get((*key, r.team), 0))
            rows["is_contract_year"].append(self.contract.get((r.player_id, int(r.season)), 0))
            dob = self.dob.get(r.player_id)
            gd = getattr(r, "gameday", None)
            rows["age_years"].append(
                round((pd.Timestamp(gd) - dob).days / 365.25, 2)
                if dob is not None and gd else np.nan)
            wx = self.weather.get(r.game_id, (np.nan, np.nan))
            rows["temp"].append(wx[0])
            rows["wind"].append(wx[1])
        for f in FEATURES:
            cands[f] = rows[f]
        return cands


def attach_neutral(cands: pd.DataFrame) -> pd.DataFrame:
    cands = cands.copy()
    for f in FEATURES:
        cands[f] = 0 if f in ("oline_outs", "is_contract_year") else np.nan
    return cands


def panel_items(lean: Dict) -> List[str]:
    items = []
    if lean.get("is_contract_year"):
        items.append("contract year (final season of latest known deal — incentive watch)")
    ol = lean.get("oline_outs")
    if ol is not None and not (isinstance(ol, float) and np.isnan(ol)) and ol >= 2:
        items.append(f"own O-line lists {int(ol)} Out/Doubtful")
    wind = lean.get("wind")
    if wind is not None and not (isinstance(wind, float) and np.isnan(wind)) and wind >= 15:
        items.append(f"wind {int(wind)} mph (passing suppressor)")
    qbc = lean.get("qb_continuity")
    if qbc is not None and not (isinstance(qbc, float) and np.isnan(qbc)) and qbc < 0.5:
        items.append("projected starting QB threw <50% of the team's trailing attempts "
                     "(usage history discount)")
    return items
