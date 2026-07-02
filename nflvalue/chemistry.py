"""Player chemistry + formation-tilt features — the TRUSTABLE generalizations.

The user's ask, triaged by data honesty (docs/decisions_p3-5.md):
exact formations/personnel and on-field matchup assignments are only free
through 2023 (NGS participation, discontinued) — unusable live. What IS
computable from play-by-play for every season, walk-forward, and live:

  shotgun_tilt_tgt    player's target share in SHOTGUN minus UNDER CENTER
                      ("LaPorta sees more work from gun") — the ML already
                      holds team_shotgun_rate, so the interaction ("DET is a
                      71% shotgun offense × his +9% gun tilt") is learnable.
  shotgun_tilt_carry  same for carries (RBs who only matter in one look).
  qb_chem_delta       share of the PROJECTED starter's career attempts that
                      target this player, minus the player's share under all
                      other QBs ("Chase-with-Burrow vs Chase-with-backup").
  key_teammate_absent this week's top same-position teammate (by trailing
                      usage) did not play — pre-game-knowable in reality
                      (inactives), approximated historically by played-flags.
  teammate_out_boost  the player's historical usage-share delta in exactly
                      those absent weeks (the Higgins-out Chase bump), as-of.
  opp_pressure_rate   opponent defense's sacks+QB-hits per dropback, rolling
                      — the live-safe slice of "defensive front/formation".

All lookups are strictly-before via the AsOfLookup pattern (the missingness
-leak guard from advanced_features): presence of a row never encodes the
current week.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .advanced_features import AsOfLookup

FEATURES = ["shotgun_tilt_tgt", "shotgun_tilt_carry", "qb_chem_delta",
            "key_teammate_absent", "teammate_out_boost", "opp_pressure_rate"]
MIN_SPLIT_PLAYS = 15     # per formation bucket before a tilt is trusted
MIN_QB_ATT = 80          # QB attempts before chemistry is trusted


def _cum_share(df: pd.DataFrame, num: str, den: str, keys: List[str]) -> pd.Series:
    """Expanding share num/den within keys, NO shift (AsOfLookup reads
    strictly-prior rows, so including the row's own week is correct)."""
    g = df.groupby(keys)
    cn = g[num].cumsum()
    cd = g[den].cumsum()
    return (cn / cd.replace(0, np.nan)).astype(float)


def build_formation_tilts(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season, week): cumulative shotgun-vs-under-center usage
    share tilt (targets and carries)."""
    plays = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)].copy()
    plays["gun"] = plays["shotgun"].fillna(0).astype(int)

    frames = []
    for role, pid_col, flag in (("tgt", "receiver_player_id", "pass"),
                                ("carry", "rusher_player_id", "rush")):
        p = plays[plays[flag] == 1].dropna(subset=[pid_col])
        wk = (p.groupby(["season", "week", "posteam", pid_col, "gun"])
              .size().rename("n").reset_index()
              .rename(columns={pid_col: "player_id"}))
        team = (p.groupby(["season", "week", "posteam", "gun"])
                .size().rename("team_n").reset_index())
        wk = wk.merge(team, on=["season", "week", "posteam", "gun"])
        wk = wk.sort_values(["player_id", "gun", "season", "week"])
        wk["share"] = _cum_share(wk, "n", "team_n", ["player_id", "gun"])
        wk["nn"] = wk.groupby(["player_id", "gun"])["n"].cumsum()
        piv = wk.pivot_table(index=["player_id", "season", "week"],
                             columns="gun", values=["share", "nn"], aggfunc="last")
        piv.columns = [f"{a}{b}" for a, b in piv.columns]
        piv = piv.reset_index().sort_values(["player_id", "season", "week"])
        piv[["share0", "share1", "nn0", "nn1"]] = (
            piv.groupby("player_id")[["share0", "share1", "nn0", "nn1"]].ffill()
            if all(c in piv.columns for c in ("share0", "share1", "nn0", "nn1"))
            else np.nan)
        ok = (piv.get("nn0", 0) >= MIN_SPLIT_PLAYS) & (piv.get("nn1", 0) >= MIN_SPLIT_PLAYS)
        piv[f"shotgun_tilt_{role}"] = np.where(
            ok, piv.get("share1", np.nan) - piv.get("share0", np.nan), np.nan)
        frames.append(piv[["player_id", "season", "week", f"shotgun_tilt_{role}"]])
    out = frames[0].merge(frames[1], on=["player_id", "season", "week"], how="outer")
    return out


def build_qb_chemistry(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (receiver, passer, season, week): cumulative share of the passer's
    attempts targeting this receiver, and the receiver's share under ALL
    OTHER passers — the delta is the chemistry signal."""
    p = pbp[(pbp["pass"] == 1)].dropna(subset=["receiver_player_id", "passer_player_id"])
    pair = (p.groupby(["season", "week", "passer_player_id", "receiver_player_id"])
            .size().rename("n").reset_index())
    qb_att = (p.groupby(["season", "week", "passer_player_id"])
              .size().rename("qb_n").reset_index())
    pair = pair.merge(qb_att, on=["season", "week", "passer_player_id"])
    pair = pair.sort_values(["receiver_player_id", "passer_player_id", "season", "week"])
    pair["cum_n"] = pair.groupby(["receiver_player_id", "passer_player_id"])["n"].cumsum()
    pair["cum_qb"] = pair.groupby(["receiver_player_id", "passer_player_id"])["qb_n"].cumsum()
    pair["share_with"] = pair["cum_n"] / pair["cum_qb"].replace(0, np.nan)
    # receiver's overall cumulative share across all passers
    tot = (pair.groupby(["receiver_player_id", "season", "week"])[["n", "qb_n"]]
           .sum().reset_index().sort_values(["receiver_player_id", "season", "week"]))
    tot["cum_all_n"] = tot.groupby("receiver_player_id")["n"].cumsum()
    tot["cum_all_qb"] = tot.groupby("receiver_player_id")["qb_n"].cumsum()
    tot["share_all"] = tot["cum_all_n"] / tot["cum_all_qb"].replace(0, np.nan)
    pair = pair.merge(tot[["receiver_player_id", "season", "week", "share_all"]],
                      on=["receiver_player_id", "season", "week"])
    pair = pair[pair["cum_qb"] >= MIN_QB_ATT]
    pair["qb_chem"] = pair["share_with"] - pair["share_all"]
    return pair.rename(columns={"receiver_player_id": "player_id",
                                "passer_player_id": "qb_id"})[
        ["player_id", "qb_id", "season", "week", "qb_chem"]]


def build_teammate_dependency(pw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Top same-position teammate (trailing usage) + the player's share delta
    in weeks that teammate missed. Returns (weekly frame, top-mate lookup)."""
    usage_col = {"WR": "targets", "TE": "targets", "RB": "carries"}
    rows = []
    for role, ucol in usage_col.items():
        d = pw[pw["role"] == role][["season", "week", "team", "player_id",
                                    "player_name", ucol, "team_pass_att",
                                    "team_rush_att"]].copy()
        d["u"] = d[ucol]
        d["den"] = d["team_pass_att"] if ucol == "targets" else d["team_rush_att"]
        d = d.sort_values(["player_id", "season", "week"])
        d["cum_u"] = d.groupby("player_id")["u"].cumsum()
        rows.append(d.assign(role=role))
    d = pd.concat(rows, ignore_index=True)

    # top teammate per (player, week): same team+role, highest trailing usage
    d["_rank"] = d.groupby(["season", "week", "team", "role"])["cum_u"] \
                  .rank(ascending=False, method="first")
    top2 = d[d["_rank"] <= 2]
    mate = top2.merge(top2, on=["season", "week", "team", "role"], suffixes=("", "_m"))
    mate = mate[mate["player_id"] != mate["player_id_m"]]
    mate["mate_absent"] = (mate["u_m"] <= 0).astype(int)
    mate["share"] = mate["u"] / mate["den"].replace(0, np.nan)
    mate = mate.sort_values(["player_id", "season", "week"])
    g = mate.groupby(["player_id", "mate_absent"])
    mate["cum_share_by_state"] = g["share"].transform(
        lambda s: s.expanding(min_periods=2).mean())
    mate["cnt_state"] = g.cumcount() + 1
    piv = mate.pivot_table(index=["player_id", "season", "week"],
                           columns="mate_absent",
                           values=["cum_share_by_state", "cnt_state"], aggfunc="last")
    piv.columns = [f"{a}{b}" for a, b in piv.columns]
    piv = piv.reset_index().sort_values(["player_id", "season", "week"])
    fill_cols = [c for c in piv.columns if c.startswith(("cum_share", "cnt_"))]
    piv[fill_cols] = piv.groupby("player_id")[fill_cols].ffill()
    ok = (piv.get("cnt_state1", 0) >= 2) & (piv.get("cnt_state0", 0) >= 4)
    piv["teammate_out_boost"] = np.where(
        ok, piv.get("cum_share_by_state1", np.nan) - piv.get("cum_share_by_state0", np.nan),
        np.nan)
    weekly_absent = mate[["player_id", "season", "week", "mate_absent", "player_id_m"]]
    return (piv[["player_id", "season", "week", "teammate_out_boost"]],
            {(r.player_id, int(r.season), int(r.week)): (int(r.mate_absent), r.player_id_m)
             for r in weekly_absent.itertuples(index=False)})


def build_pressure(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defense sacks+QB-hits per dropback, cumulative (as-of consumed)."""
    p = pbp[pbp["pass"] == 1].copy()
    wk = (p.groupby(["season", "week", "defteam"])
          .agg(pres=("sack", "sum"), hits=("qb_hit", "sum"), n=("pass", "sum"))
          .reset_index().sort_values(["defteam", "season", "week"]))
    g = wk.groupby("defteam")
    wk["opp_pressure_rate"] = ((g["pres"].cumsum() + g["hits"].cumsum())
                               / g["n"].cumsum().replace(0, np.nan))
    return wk.rename(columns={"defteam": "team"})[
        ["team", "season", "week", "opp_pressure_rate"]]


class ChemistryPack:
    def __init__(self, pbp: Optional[pd.DataFrame] = None,
                 pw: Optional[pd.DataFrame] = None,
                 schedules: Optional[pd.DataFrame] = None):
        if pbp is None:
            from .advanced_features import load_pbp_ext
            pbp = load_pbp_ext()
        tilts = build_formation_tilts(pbp)
        self.tilts = AsOfLookup(tilts, ["shotgun_tilt_tgt", "shotgun_tilt_carry"])
        chem = build_qb_chemistry(pbp)
        self.chem: Dict[Tuple[str, str], AsOfLookup] = {}
        for qb, grp in chem.groupby("qb_id"):
            self.chem[qb] = AsOfLookup(grp, ["qb_chem"])
        dep, self.mate_state = (build_teammate_dependency(pw)
                                if pw is not None else (pd.DataFrame(
                                    columns=["player_id", "season", "week",
                                             "teammate_out_boost"]), {}))
        self.dep = AsOfLookup(dep, ["teammate_out_boost"])
        pres = build_pressure(pbp).rename(columns={"team": "player_id"})
        self.pressure = AsOfLookup(pres, ["opp_pressure_rate"])
        # projected starter per (season, week, team) from schedules
        self.starter: Dict[Tuple, str] = {}
        if schedules is not None:
            for g in schedules[schedules["game_type"] == "REG"].itertuples(index=False):
                for team, qb in ((g.home_team, getattr(g, "home_qb_id", None)),
                                 (g.away_team, getattr(g, "away_qb_id", None))):
                    if isinstance(qb, str) and qb:
                        self.starter[(int(g.season), int(g.week), team)] = qb

    def attach(self, cands: pd.DataFrame,
               out_player_ids: Optional[set] = None) -> pd.DataFrame:
        cands = cands.copy()
        rows = {f: [] for f in FEATURES}
        for r in cands.itertuples(index=False):
            key = (int(r.season), int(r.week))
            t = self.tilts.get(r.player_id, *key)
            rows["shotgun_tilt_tgt"].append(t[0])
            rows["shotgun_tilt_carry"].append(t[1])
            qb = self.starter.get((*key, r.team))
            chem = self.chem.get(qb).get(r.player_id, *key) if qb in self.chem else (np.nan,)
            rows["qb_chem_delta"].append(chem[0])
            rows["teammate_out_boost"].append(self.dep.get(r.player_id, *key)[0])
            state = self.mate_state.get((r.player_id, *key))
            if out_player_ids is not None and state is not None:
                absent = int(state[1] in out_player_ids)      # live: availability says OUT
            else:
                absent = state[0] if state is not None else 0  # historical: played-flag
            rows["key_teammate_absent"].append(absent)
            rows["opp_pressure_rate"].append(self.pressure.get(r.defteam, *key)[0])
        for f in FEATURES:
            cands[f] = rows[f]
        return cands


def attach_neutral(cands: pd.DataFrame) -> pd.DataFrame:
    cands = cands.copy()
    for f in FEATURES:
        cands[f] = 0 if f == "key_teammate_absent" else np.nan
    return cands


def panel_items(lean: Dict) -> List[str]:
    items = []
    tilt = lean.get("shotgun_tilt_tgt")
    if tilt is not None and not (isinstance(tilt, float) and np.isnan(tilt)) and abs(tilt) >= 0.05:
        items.append(f"formation tilt: {'+' if tilt > 0 else ''}{tilt:.0%} target share "
                     f"in shotgun vs under center")
    chem = lean.get("qb_chem_delta")
    if chem is not None and not (isinstance(chem, float) and np.isnan(chem)) and abs(chem) >= 0.04:
        items.append(f"QB chemistry: {'+' if chem > 0 else ''}{chem:.0%} of the projected "
                     "starter's attempts vs his rate under other QBs")
    if lean.get("key_teammate_absent"):
        boost = lean.get("teammate_out_boost")
        extra = (f" (historical bump {'+' if boost > 0 else ''}{boost:.0%} share)"
                 if boost is not None and not (isinstance(boost, float) and np.isnan(boost))
                 else "")
        items.append("top position-mate absent this week" + extra)
    return items
