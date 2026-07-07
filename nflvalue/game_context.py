"""Observation-quality / game-context tags for the player-week panel.

Two of the "is this stat line even representative?" objections, turned into
measurable per-observation tags so the trailing means (and, later, calibration
/ correlation / the ML frame) can DOWN-WEIGHT or drop a contaminated game
instead of letting it silently nerf a projection:

  1. INJURY-SHORTENED weeks -- a player whose game was truncated: the
     ``early_exit`` pbp signature (meaningful first-half usage, nothing after
     halftime while his team kept playing) OR an anomalous collapse in snap
     share versus his own trailing norm. A 1-catch line from a guy who tore a
     hamstring on the opening drive is not evidence about his role.

  2. REST / MEANINGLESS games -- late-season games a team enters with its
     playoff fate effectively settled (a seed clinched, or eliminated), where
     starters get rested or capped. There is no free official clinch feed, so
     this is an explicit, coarse PROXY built from records-to-date and a
     conference-cut approximation (no tiebreakers). Labeled as a proxy
     everywhere it surfaces.

DESIGN NOTES
------------
* Leakage: these tags LABEL a game using only that game's own data (or, for the
  snap-collapse arm, the player's STRICTLY-PRIOR trailing snap share). They
  become leak-safe *inputs* because the cleaner only ever applies a prior-week
  tag when projecting a later week -- identical discipline to every ``roll_*``
  feature. The sweep and any production cleaner must consume prior-week tags
  only; that is asserted in the sweep, not assumed here.
* Nothing here is scored or wired into production means yet. This is the shared
  primitive that ``scripts/fit_recency_weight.py`` uses to test whether
  clean-then-weight beats the status quo, under the same measured-verdict rule
  the rest of the codebase lives by.
* Blowout GARBAGE TIME is deliberately NOT re-implemented here: it was measured
  and REJECTED in Phase 6.3 (no MAE improvement; receiving 24.77 vs 24.58,
  receptions 1.670 vs 1.658 filtered-vs-not). That play-level machinery stays
  in ``features.py`` behind ``GARBAGE_FILTER_ENABLED``. Injury-shortened and
  rest/meaningless cleaning are week-level and, unlike garbage time, untested.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Static AFC/NFC map (stable; relocations keep their abb: OAK->LV, SD->LAC,
# STL->LA all pre-2019 base window). Used only for the meaningless-game cut
# approximation; a missing/unknown team simply never gets flagged.
CONFERENCE = {
    # AFC
    "BUF": "AFC", "MIA": "AFC", "NE": "AFC", "NYJ": "AFC",
    "BAL": "AFC", "CIN": "AFC", "CLE": "AFC", "PIT": "AFC",
    "HOU": "AFC", "IND": "AFC", "JAX": "AFC", "TEN": "AFC",
    "DEN": "AFC", "KC": "AFC", "LV": "AFC", "LAC": "AFC",
    # NFC
    "DAL": "NFC", "NYG": "NFC", "PHI": "NFC", "WAS": "NFC",
    "CHI": "NFC", "DET": "NFC", "GB": "NFC", "MIN": "NFC",
    "ATL": "NFC", "CAR": "NFC", "NO": "NFC", "TB": "NFC",
    "ARI": "NFC", "LA": "NFC", "SF": "NFC", "SEA": "NFC",
}
PLAYOFF_SEEDS = 7  # per conference, 2020+. (6 pre-2020; the extra seed only
                   # makes the "eliminated"/"clinched" proxy MORE conservative,
                   # so we use 7 across the board -- fewer false flags.)


# --------------------------------------------------------------------------- #
# 1. Injury-shortened weeks
# --------------------------------------------------------------------------- #
def _early_exit_signature(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, player_id) ``early_exit`` 0/1 -- the same signature
    as ``features._early_exit_week`` (meaningful H1 usage, zero H2 while the
    team ran >=10 H2 plays), reimplemented here to keep this module free of the
    heavy features import. Kept in lockstep by ``tests/test_game_context.py``.
    """
    need = {"qtr", "season", "week", "posteam"}
    if not need <= set(pbp.columns):
        return pd.DataFrame(columns=["season", "week", "player_id", "early_exit"])
    frames = []
    for id_col in ("receiver_player_id", "rusher_player_id", "passer_player_id"):
        if id_col not in pbp.columns:
            continue
        d = pbp.dropna(subset=[id_col])
        frames.append(pd.DataFrame({
            "season": d["season"], "week": d["week"], "player_id": d[id_col],
            "posteam": d["posteam"],
            "h1": (d["qtr"] <= 2).astype(float), "h2": (d["qtr"] >= 3).astype(float)}))
    if not frames:
        return pd.DataFrame(columns=["season", "week", "player_id", "early_exit"])
    u = pd.concat(frames, ignore_index=True)
    per = u.groupby(["season", "week", "player_id", "posteam"])[["h1", "h2"]].sum().reset_index()
    team_h2 = (pbp[pbp["qtr"] >= 3].groupby(["season", "week", "posteam"])
               .size().rename("team_h2_plays").reset_index())
    per = per.merge(team_h2, on=["season", "week", "posteam"], how="left")
    per["early_exit"] = ((per["h1"] >= 3) & (per["h2"] == 0)
                         & (per["team_h2_plays"].fillna(0) >= 10)).astype(float)
    return per[["season", "week", "player_id", "early_exit"]].drop_duplicates(
        subset=["season", "week", "player_id"])


def injury_shortened_weeks(pbp: pd.DataFrame,
                           snap_counts: Optional[pd.DataFrame] = None,
                           snap_drop: float = 0.5,
                           snap_min_history: int = 3) -> pd.DataFrame:
    """One row per (season, week, player_id) with ``injury_shortened`` 0/1 and a
    ``reason`` string.

    Two independent arms (OR'd):
      * ``early_exit`` -- the pbp truncation signature (self-contained to the
        week).
      * ``snap_collapse`` -- offense snap share fell below ``snap_drop`` x the
        player's STRICTLY-PRIOR trailing-median share (needs
        ``snap_min_history`` prior games). Requires ``snap_counts`` with
        columns [season, week, pfr_player_id|player_id, offense_pct]; skipped
        silently if absent, so callers without the snaps cache still work.
    """
    ee = _early_exit_signature(pbp)
    ee = ee[ee["early_exit"] == 1.0][["season", "week", "player_id"]].assign(_ee=True) \
        if len(ee) else pd.DataFrame(columns=["season", "week", "player_id", "_ee"])

    sc_flags = pd.DataFrame(columns=["season", "week", "player_id", "_snap"])
    if snap_counts is not None and len(snap_counts):
        sc = snap_counts.copy()
        id_col = "player_id" if "player_id" in sc.columns else (
            "pfr_player_id" if "pfr_player_id" in sc.columns else None)
        if id_col is not None and "offense_pct" in sc.columns:
            sc = sc.rename(columns={id_col: "player_id"})
            sc = sc[["season", "week", "player_id", "offense_pct"]].dropna(subset=["player_id"])
            sc = sc.sort_values(["player_id", "season", "week"])
            # strictly-prior trailing median share (shift(1) then rolling) --
            # leak-safe baseline, mirrors the roll_* idiom
            g = sc.groupby("player_id")["offense_pct"]
            base = g.transform(lambda s: s.shift(1).rolling(8, min_periods=snap_min_history).median())
            sc["_snap"] = (sc["offense_pct"] < snap_drop * base) & base.notna()
            sc_flags = sc[sc["_snap"]][["season", "week", "player_id"]].assign(_snap=True)

    out = pd.merge(ee, sc_flags, on=["season", "week", "player_id"], how="outer")
    out["_ee"] = out["_ee"].eq(True)      # NaN (unmatched) -> False, no object-fillna warning
    out["_snap"] = out["_snap"].eq(True)
    out["injury_shortened"] = (out["_ee"] | out["_snap"]).astype(float)
    out["reason"] = np.where(out["_ee"] & out["_snap"], "early_exit+snap_collapse",
                             np.where(out["_ee"], "early_exit", "snap_collapse"))
    return out[["season", "week", "player_id", "injury_shortened", "reason"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Rest / meaningless-game proxy (records-to-date + conference-cut approx)
# --------------------------------------------------------------------------- #
def records_to_date(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team): wins/losses/games_played STRICTLY BEFORE that
    week, plus games_left in the regular season. Leak-safe by construction
    (shift the cumulative count). REG games only."""
    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"].isin(["REG", "REGULAR", None]) | s["game_type"].isna()]
    long = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({
            "season": s["season"], "week": s["week"], "team": s[f"{side}_team"],
            "pf": s[f"{side}_score"], "pa": s[f"{opp}_score"]})
        long.append(d)
    g = pd.concat(long, ignore_index=True).dropna(subset=["team"])
    g["played"] = g["pf"].notna().astype(int)
    g["win"] = (g["pf"] > g["pa"]).astype(int) * g["played"]
    g["loss"] = (g["pf"] < g["pa"]).astype(int) * g["played"]
    g = g.sort_values(["season", "team", "week"])
    grp = g.groupby(["season", "team"], group_keys=False)
    # STRICTLY-prior: shift the cumulative sums so week W sees only weeks < W
    g["wins"] = grp["win"].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
    g["losses"] = grp["loss"].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
    g["games_played"] = grp["played"].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
    total = g.groupby(["season", "team"])["played"].transform("sum")  # season length seen
    g["games_left"] = (total - g["games_played"]).clip(lower=0)
    return g[["season", "week", "team", "wins", "losses", "games_played", "games_left"]]


def meaningless_game_flags(schedules: pd.DataFrame, week_min: int = 17,
                           clear_margin: float = 2.0) -> pd.DataFrame:
    """COARSE PROXY. One row per (season, week, team) with ``meaningless`` 0/1
    and ``reason`` in {clinched, eliminated, ""}.

    From records-to-date, within each (season, week, conference): approximate
    the playoff cut as the ``PLAYOFF_SEEDS``-th team's wins. A team is flagged
    only in weeks >= ``week_min`` and only when the race is out of reach by
    ``clear_margin`` even if every remaining game broke the other way:
      * clinched   -- wins - cut_wins > games_left + clear_margin
      * eliminated -- (cut_wins - wins) > games_left + clear_margin
    No tiebreakers, no division logic -- deliberately conservative so a flag is
    a strong signal, at the cost of missing genuine rest cases. The sweep varies
    ``week_min``/``clear_margin`` to test sensitivity; do not treat as truth.
    """
    rec = records_to_date(schedules)
    rec = rec[rec["week"] >= week_min].copy()
    if rec.empty:
        return pd.DataFrame(columns=["season", "week", "team", "meaningless", "reason"])
    rec["conf"] = rec["team"].map(CONFERENCE)
    rec = rec.dropna(subset=["conf"])
    out = []
    for (season, week, conf), grp in rec.groupby(["season", "week", "conf"]):
        wins_sorted = grp["wins"].sort_values(ascending=False).to_numpy()
        if len(wins_sorted) >= PLAYOFF_SEEDS:
            cut = wins_sorted[PLAYOFF_SEEDS - 1]
        else:
            cut = wins_sorted[-1] if len(wins_sorted) else 0.0
        for r in grp.itertuples(index=False):
            gl = r.games_left
            clinched = (r.wins - cut) > (gl + clear_margin)
            eliminated = (cut - r.wins) > (gl + clear_margin)
            flag = 1.0 if (clinched or eliminated) else 0.0
            reason = "clinched" if clinched else ("eliminated" if eliminated else "")
            out.append({"season": r.season, "week": r.week, "team": r.team,
                        "meaningless": flag, "reason": reason})
    return pd.DataFrame(out, columns=["season", "week", "team", "meaningless", "reason"])


# --------------------------------------------------------------------------- #
# Convenience: stamp both tags onto a player-week frame
# --------------------------------------------------------------------------- #
def tag_player_weeks(pw: pd.DataFrame, pbp: Optional[pd.DataFrame] = None,
                     schedules: Optional[pd.DataFrame] = None,
                     snap_counts: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return ``pw`` with two added 0/1 columns: ``injury_shortened`` (from
    pbp/snaps) and ``game_meaningless`` (from schedules, by the player's team).
    Missing inputs leave the corresponding column all-zero. Non-destructive."""
    out = pw.copy()
    out["injury_shortened"] = 0.0
    out["game_meaningless"] = 0.0
    if pbp is not None and len(pbp):
        inj = injury_shortened_weeks(pbp, snap_counts=snap_counts)
        if len(inj):
            out = out.drop(columns=["injury_shortened"]).merge(
                inj[["season", "week", "player_id", "injury_shortened"]],
                on=["season", "week", "player_id"], how="left")
            out["injury_shortened"] = out["injury_shortened"].fillna(0.0)
    if schedules is not None and len(schedules) and "team" in out.columns:
        mg = meaningless_game_flags(schedules)
        if len(mg):
            out = out.drop(columns=["game_meaningless"]).merge(
                mg[["season", "week", "team", "meaningless"]].rename(
                    columns={"meaningless": "game_meaningless"}),
                on=["season", "week", "team"], how="left")
            out["game_meaningless"] = out["game_meaningless"].fillna(0.0)
    return out
