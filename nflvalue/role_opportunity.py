"""Current-season vs historical-prior pooling for player opportunity and efficiency.

Pure computation, no I/O: the caller supplies every frame and an explicit
timezone-aware ``as_of``.  Protocol: ``analysis/role_opportunity_protocol.json``
(frozen before evaluation); results: ``specialist_artifacts/CURRENT_VS_PRIOR.md``.

For each player and quantity (target/carry/pass share of team volume, team
pass/rush attempts per game, and per-opportunity efficiency) the forecast is

    prior     = historical prior: the player's season S-1 ratio, shrunk toward
                the S-1 position mean with ``k0`` (or, with no S-1 games, the
                mean of first-appearance player-seasons at the position).
                It is a *historical prior*, not a pre-season projection --
                no pre-season projection exists in the data.
    current   = season-S ratio from games on the CURRENT team strictly before
                the target week (sum numerator / sum denominator).  No
                season-S game -> ``None`` (missing, not zero).
    posterior = (Y_cur + k * prior) / (D_cur + k)
    w_current = D_cur / (D_cur + k)

``k`` is a prior strength in DENOMINATOR units (team attempts for shares,
player opportunities for efficiency, games for team volume), learned by
``fit_hyperparameters`` on earlier seasons only as ``phi / rho``: ``phi`` is a
game-cluster sampling dispersion (games, not plays, are the independent unit)
and ``rho`` the scaled variance of a season's true ratio around its prior.
It differs by quantity, position and regime (same_team / team_changed /
no_prior) -- a team change widens prior uncertainty (smaller ``k``); it never
adds a performance boost.  There is no fixed "N games = X%" rule.

Routes, snaps and active/inactive status are not in the supplied data; they
are reported ``unavailable`` and targets are never relabelled as routes.
Absent-teammate reallocation stays with ``nflvalue.availability`` and is not
repeated here; this module only conserves team shares (sum <= 1).

Input schema -- ``games`` (one or more rows per player-game in which the
player recorded a pass attempt, target or carry; split rows of one game are
summed, team attempts counted once per game):
    season:int, week:int, game_id:str, player_id:str (gsis), team:str,
    position:str in {QB,RB,WR,TE}, targets, receptions, rec_yards, carries,
    rush_yards, pass_attempts, pass_yards, team_pass_att, team_rush_att,
    optional game_start (tz-aware timestamp; rows at/after ``as_of`` dropped).
``targets`` (players to forecast): player_id, team, position, game_id,
    optional game_start (``as_of`` must be before it).

Integration recipe (see specialist_artifacts/FINAL.md):
    games = player_games_from_player_week(pw)            # adapter, pure
    hyper = fit_hyperparameters(games, before_season=S)  # earlier seasons only
    out = forecast_opportunity(games, targets, season=S, week=W, as_of=as_of, hyper=hyper)
    volume = out["players"][i]["expected_targets"]       # replaces team roll x roll share
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

MODEL_VERSION = "role-opportunity-pooling-v1"

#: (numerator, denominator, scale function) per player quantity.
PLAYER_QUANTITIES: Dict[str, tuple] = {
    "target_share": ("targets", "team_pass_att", "prop"),
    "carry_share": ("carries", "team_rush_att", "prop"),
    "pass_share": ("pass_attempts", "team_pass_att", "prop"),
    "ypt": ("rec_yards", "targets", "sq"),
    "catch_rate": ("receptions", "targets", "prop"),
    "ypc": ("rush_yards", "carries", "sq"),
    "ypa": ("pass_yards", "pass_attempts", "sq"),
}
QUANTITY_POSITIONS: Dict[str, tuple] = {
    "target_share": ("WR", "TE", "RB"),
    "carry_share": ("RB",),
    "pass_share": ("QB",),
    "ypt": ("WR", "TE", "RB"),
    "catch_rate": ("WR", "TE", "RB"),
    "ypc": ("RB",),
    "ypa": ("QB",),
}
OPPORTUNITY_QUANTITIES = ("target_share", "carry_share", "pass_share")
EFFICIENCY_QUANTITIES = ("ypt", "catch_rate", "ypc", "ypa")
TEAM_QUANTITIES = ("team_pass_att", "team_rush_att")
#: share -> (team volume it multiplies, expected-volume output name)
SHARE_VOLUME = {"target_share": ("team_pass_att", "expected_targets"),
                "carry_share": ("team_rush_att", "expected_carries"),
                "pass_share": ("team_pass_att", "expected_pass_attempts")}
REGIMES = ("same_team", "team_changed", "no_prior")
MIN_CELL = 30
K_BOUNDS = (1e-3, 1e5)
RHO_FLOOR = 1e-6
G_FLOOR = 1e-4
STAT_COLS = ["targets", "receptions", "rec_yards", "carries", "rush_yards",
             "pass_attempts", "pass_yards"]
TEAM_COLS = ["team_pass_att", "team_rush_att"]
UNITS = {"target_share": "share of team pass attempts", "carry_share": "share of team rush attempts",
         "pass_share": "share of team pass attempts", "ypt": "yards per target",
         "catch_rate": "receptions per target", "ypc": "yards per carry", "ypa": "yards per attempt",
         "team_pass_att": "pass attempts per game", "team_rush_att": "rush attempts per game",
         "expected_targets": "targets", "expected_carries": "carries",
         "expected_pass_attempts": "pass attempts"}


def _g(p: float, kind: str) -> float:
    if kind == "prop":
        return max(p * (1.0 - p), G_FLOOR)
    if kind == "sq":
        return max(p * p, G_FLOOR)
    return 1.0


# --------------------------------------------------------------------------- #
# Per-game tables
# --------------------------------------------------------------------------- #
def _player_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, game): stats summed, team attempts counted once."""
    keys = ["season", "week", "game_id", "player_id", "team"]
    agg = {c: "sum" for c in STAT_COLS}
    agg.update({c: "max" for c in TEAM_COLS})
    agg["position"] = "first"
    return games.groupby(keys, as_index=False, sort=False).agg(agg)


def _team_games(pg: pd.DataFrame) -> pd.DataFrame:
    return (pg.groupby(["season", "week", "game_id", "team"], as_index=False, sort=False)[TEAM_COLS]
            .max())


def _main_team(pg: pd.DataFrame) -> pd.Series:
    """player_id -> team with the most games (ties: the latest week)."""
    c = (pg.groupby(["player_id", "team"]).agg(n=("game_id", "nunique"), last=("week", "max"))
         .reset_index().sort_values(["player_id", "n", "last"]))
    return c.groupby("player_id")["team"].last()


# --------------------------------------------------------------------------- #
# Hyperparameters (earlier seasons only)
# --------------------------------------------------------------------------- #
def _moment_rho(r, prior, d, phi, kind) -> float:
    vals = [((ri - pi) ** 2 - phi * _g(pi, kind) / di) / _g(pi, kind)
            for ri, pi, di in zip(r, prior, d)]
    return max(float(np.mean(vals)), RHO_FLOOR) if vals else float("nan")


def _cell(phi, rho, mu, mu_new, k0, n, pooled):
    k = float(np.clip(phi / rho, *K_BOUNDS))
    return dict(k=k, rho=float(rho), phi=float(phi), mu=float(mu), mu_new=float(mu_new),
                k0=float(k0), n_player_seasons=int(n), pooled=bool(pooled))


def _fit_ratio(units: pd.DataFrame, prev: Optional[pd.DataFrame], kind: str) -> Dict[str, dict]:
    """``units``: Y, D, n, SS, season, id, team, has_prev_season_data.
    ``prev``: indexed by (id, season) of the FOLLOWING season with Y_prev,
    D_prev, team_prev.  Returns regime -> cell."""
    u = units[units["D"] > 0].copy()
    u["r"] = u["Y"] / u["D"]
    multi = u[u["n"] >= 2]
    den = float(sum(_g(r, kind) * d for r, d in zip(multi["r"], multi["D"])))
    phi = float((multi["n"] / (multi["n"] - 1) * multi["SS"]).sum() / den) if den > 0 else float("nan")
    mu = float(u["r"].mean())
    rho0 = _moment_rho(u["r"], [mu] * len(u), u["D"], phi, kind)
    k0 = float(np.clip(phi / rho0, *K_BOUNDS))

    pair = u[u["has_prev_season_data"]].copy()
    pair = pair.merge(prev, on=["id", "season"], how="left")
    has = pair["D_prev"].notna()
    new = pair[~has]
    mu_new = float(new["r"].mean()) if len(new) else mu
    pair["prior"] = np.where(has, (pair["Y_prev"].fillna(0) + k0 * mu) / (pair["D_prev"].fillna(0) + k0), mu_new)
    pair["regime"] = np.where(~has, "no_prior",
                              np.where(pair["team_prev"] == pair["team"], "same_team", "team_changed"))
    rho_all = _moment_rho(pair["r"], pair["prior"], pair["D"], phi, kind)
    out = {}
    for reg in REGIMES:
        c = pair[pair["regime"] == reg]
        if len(c) >= MIN_CELL:
            out[reg] = _cell(phi, _moment_rho(c["r"], c["prior"], c["D"], phi, kind), mu, mu_new, k0, len(c), False)
        else:
            out[reg] = _cell(phi, rho_all, mu, mu_new, k0, len(c), True)
    return out


def fit_hyperparameters(games: pd.DataFrame, *, before_season: int) -> dict:
    """Learn ``k`` per quantity x position x regime from seasons < ``before_season``.

    Method of moments (protocol ``k_estimation``).  Rows from ``before_season``
    or later are dropped before anything is computed.
    """
    g = games[games["season"] < before_season]
    pg = _player_games(g)
    seasons = sorted(int(s) for s in pg["season"].unique())
    out = {"version": MODEL_VERSION, "fit_seasons": seasons, "before_season": int(before_season),
           "quantities": {}}
    if not seasons:
        return out
    season_set = set(seasons)
    main_by_season = (pg.groupby(["player_id", "season", "team"])
                      .agg(n=("game_id", "nunique"), last=("week", "max")).reset_index()
                      .sort_values(["player_id", "season", "n", "last"])
                      .groupby(["player_id", "season"])["team"].last()
                      .rename("team_prev").reset_index())

    for q, (num, den, kind) in PLAYER_QUANTITIES.items():
        out["quantities"][q] = {}
        for pos in QUANTITY_POSITIONS[q]:
            sub = pg[pg["position"] == pos]
            if sub.empty:
                continue
            u = sub.groupby(["player_id", "season", "team"]).agg(
                Y=(num, "sum"), D=(den, "sum"), n=("game_id", "nunique")).reset_index()
            r = (u["Y"] / u["D"].replace(0, np.nan)).rename("r_ps")
            sub = sub.merge(u.assign(r_ps=r)[["player_id", "season", "team", "r_ps"]],
                            on=["player_id", "season", "team"])
            sub["_sq"] = (sub[num] - sub["r_ps"].fillna(0) * sub[den]) ** 2
            ss = sub.groupby(["player_id", "season", "team"])["_sq"].sum().rename("SS").reset_index()
            u = u.merge(ss, on=["player_id", "season", "team"]).rename(columns={"player_id": "id"})
            u["has_prev_season_data"] = u["season"].sub(1).isin(season_set)
            # previous-season totals come from ALL of the player's rows (any position)
            allp = pg.groupby(["player_id", "season"]).agg(Y_prev=(num, "sum"), D_prev=(den, "sum"))
            allp = allp.reset_index()
            allp = allp.merge(main_by_season, on=["player_id", "season"])
            allp["season"] = allp["season"] + 1
            allp = allp.rename(columns={"player_id": "id"})
            out["quantities"][q][pos] = _fit_ratio(u, allp, kind)

    tg = _team_games(pg)
    for q in TEAM_QUANTITIES:
        u = tg.groupby(["team", "season"]).agg(Y=(q, "sum"), n=("game_id", "nunique")).reset_index()
        u["D"] = u["n"].astype(float)
        tg2 = tg.merge(u.assign(m=u["Y"] / u["D"])[["team", "season", "m"]], on=["team", "season"])
        tg2["_sq"] = (tg2[q] - tg2["m"]) ** 2
        u = u.merge(tg2.groupby(["team", "season"])["_sq"].sum().rename("SS").reset_index(),
                    on=["team", "season"]).rename(columns={"team": "id"})
        u["team"] = u["id"]
        u["has_prev_season_data"] = u["season"].sub(1).isin(season_set)
        prev = u[["id", "season", "Y", "D"]].rename(columns={"Y": "Y_prev", "D": "D_prev"}).copy()
        prev["season"] = prev["season"] + 1
        prev["team_prev"] = prev["id"]
        cells = _fit_ratio(u, prev, "unit")
        out["quantities"][q] = {"TEAM": {"team": cells["same_team"]}}
    return out


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
def _blend(y_cur, d_cur, n_cur, prior, cell, kind) -> dict:
    k = cell["k"]
    post = (y_cur + k * prior) / (d_cur + k)
    var = cell["rho"] * _g(prior, kind) * k / (k + d_cur)
    return dict(current_estimate=(y_cur / d_cur) if d_cur > 0 else None,
                current_games=int(n_cur), current_denominator=float(d_cur), k=float(k),
                w_current=float(d_cur / (d_cur + k)), posterior=float(post),
                posterior_sd=float(math.sqrt(max(var, 0.0))))


def _check_as_of(as_of) -> pd.Timestamp:
    if not isinstance(as_of, dt.datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be a timezone-aware datetime")
    return pd.Timestamp(as_of).tz_convert("UTC")


def forecast_opportunity(games: pd.DataFrame, targets: pd.DataFrame, *, season: int, week: int,
                         as_of: dt.datetime, hyper: dict, consumption: str = "shadow",
                         regime_events: Optional[Iterable[dict]] = None,
                         source: Optional[dict] = None) -> dict:
    """Prior, current, weights and posterior opportunity for ``targets`` at (season, week).

    Pure: inputs are not mutated and nothing is fetched.  ``consumption`` is
    "shadow" (default) or "primary"; only a primary consumer may mark factor
    records ``numeric_applied``.  ``regime_events`` (e.g. a verified
    starter/backup switch from a news adapter: player_id, event, verified,
    source_id, published_at) are recorded as context only -- no numeric
    mapping for them has been tested.
    """
    ts_as_of = _check_as_of(as_of)
    if consumption not in ("shadow", "primary"):
        raise ValueError("consumption must be 'shadow' or 'primary'")
    if "game_start" in targets.columns:
        ks = pd.to_datetime(targets["game_start"], utc=True)
        if (ks.notna() & (ks <= ts_as_of)).any():
            raise ValueError("as_of is at or after a target game's kickoff")
    source = dict(source or {})
    src_id = source.get("source_id", "caller_supplied_games_frame")

    g = games
    before_week = (g["season"] < season) | ((g["season"] == season) & (g["week"] < week))
    n_week = int((~before_week).sum())
    keep = before_week
    n_clock = 0
    if "game_start" in g.columns:
        gs = pd.to_datetime(g["game_start"], utc=True)
        late = before_week & gs.notna() & (gs >= ts_as_of)
        n_clock = int(late.sum())
        keep = keep & ~late
    pg = _player_games(g[keep])
    tg = _team_games(pg)
    prev = pg[pg["season"] == season - 1]
    cur = pg[pg["season"] == season]
    prev_main = _main_team(prev) if not prev.empty else pd.Series(dtype=object)
    prev_tot = prev.groupby("player_id")[STAT_COLS + TEAM_COLS].sum()
    prev_n = prev.groupby("player_id")["game_id"].nunique()

    teams: Dict[str, dict] = {}
    for team in sorted(set(targets["team"])):
        teams[team] = {}
        for q in TEAM_QUANTITIES:
            cell = hyper["quantities"][q]["TEAM"]["team"]
            tp = tg[(tg["team"] == team) & (tg["season"] == season - 1)]
            tc = tg[(tg["team"] == team) & (tg["season"] == season)]
            n_prev = tp["game_id"].nunique()
            prior = (tp[q].sum() + cell["k0"] * cell["mu"]) / (n_prev + cell["k0"])
            rec = dict(prior_estimate=float(prior), prior_games=int(n_prev),
                       prior_source="historical_prior_prev_season" if n_prev else "league_mean_no_prior",
                       regime="team")
            rec.update(_blend(float(tc[q].sum()), float(tc["game_id"].nunique()),
                              tc["game_id"].nunique(), prior, cell, "unit"))
            teams[team][q] = rec

    players: List[dict] = []
    for t in targets.to_dict("records"):
        pid, team, pos = t["player_id"], t["team"], t["position"]
        mine = cur[(cur["player_id"] == pid) & (cur["team"] == team)]
        other = cur[(cur["player_id"] == pid) & (cur["team"] != team)]
        has_prev = pid in prev_n.index
        regime = ("no_prior" if not has_prev
                  else "same_team" if prev_main.get(pid) == team else "team_changed")
        rec = dict(player_id=pid, team=team, position=pos, game_id=t.get("game_id"),
                   season=int(season), week=int(week), regime=regime,
                   prior_season_team=prev_main.get(pid) if has_prev else None,
                   same_season_other_team_games=int(other["game_id"].nunique()))
        for q, (num, den, kind) in PLAYER_QUANTITIES.items():
            cells = hyper["quantities"].get(q, {}).get(pos)
            if not cells:
                rec[q] = None
                continue
            cell = cells[regime]
            if has_prev:
                y_p, d_p = float(prev_tot.at[pid, num]), float(prev_tot.at[pid, den])
                prior = (y_p + cell["k0"] * cell["mu"]) / (d_p + cell["k0"])
                src, n_p = "historical_prior_prev_season", int(prev_n.at[pid])
            else:
                d_p, prior, src, n_p = 0.0, cell["mu_new"], "position_mean_no_prior", 0
            r = dict(prior_estimate=float(prior), prior_games=n_p, prior_denominator=d_p,
                     prior_source=src, regime=regime, pooled_cell=cell["pooled"])
            r.update(_blend(float(mine[num].sum()), float(mine[den].sum()),
                            mine["game_id"].nunique(), prior, cell, kind))
            rec[q] = r
        players.append(rec)

    # team-share conservation (sum of shares over supplied players <= 1)
    for team, tv in teams.items():
        for q in OPPORTUNITY_QUANTITIES:
            recs = [p[q] for p in players if p["team"] == team and p.get(q)]
            raw = float(sum(r["posterior"] for r in recs))
            scale = 1.0 / raw if raw > 1.0 else 1.0
            tv[f"{q}_raw_sum"], tv[f"{q}_scale"] = raw, scale
            tv[f"{q}_unallocated"] = max(0.0, 1.0 - raw)
            for r in recs:
                r["posterior_conserved"] = r["posterior"] * scale
    for p in players:
        for q, (vol, name) in SHARE_VOLUME.items():
            s = p.get(q)
            if s is None:
                p[name] = p[f"{name}_prior_only"] = p[f"{name}_unconserved"] = None
                continue
            tv = teams[p["team"]][vol]
            p[name] = tv["posterior"] * s["posterior_conserved"]
            # conservation is only as good as the supplied player set; with no
            # confirmed pre-game active list (e.g. three rostered QBs) use this
            p[f"{name}_unconserved"] = tv["posterior"] * s["posterior"]
            p[f"{name}_prior_only"] = tv["prior_estimate"] * s["prior_estimate"]

    factors = _factor_records(players, teams, ts_as_of, consumption, regime_events, src_id,
                              bool(source.get("verified", False)))
    return {"version": MODEL_VERSION, "players": players, "teams": teams, "factors": factors,
            "meta": {"season": int(season), "week": int(week), "as_of": ts_as_of.isoformat(),
                     "hyper_version": hyper.get("version"), "hyper_fit_seasons": hyper.get("fit_seasons"),
                     "rows_supplied": int(len(g)), "rows_used": int(keep.sum()),
                     "rows_excluded_by_week": n_week, "rows_excluded_by_clock": n_clock,
                     "consumption": consumption, "source": source}}


# --------------------------------------------------------------------------- #
# Factor evidence records (shared contract)
# --------------------------------------------------------------------------- #
def _base(fid, cat, eid, etype, game_id, as_of, src_id, verified):
    return dict(factor_id=fid, category=cat, entity_id=eid, entity_type=etype, game_id=game_id,
                as_of=as_of.isoformat(), observation=None, value=None, unit=None,
                observed_at=None, published_at=None, fetched_at=None, source_url=None,
                source_id=src_id, verified=verified, cutoff_ok=True, measurement_kind="projected",
                status="shadow_only", model_version=MODEL_VERSION, component="role_opportunity",
                feature_name=None, consumed=False, numerical_effect=None, numerical_effect_unit=None,
                numerical_effect_method=None, support_games=None, support_opportunities=None,
                reason_not_applied=None, rationale="", uncertainty=None)


def _factor_records(players, teams, as_of, consumption, regime_events, src_id, verified):
    applied = consumption == "primary"
    cf = "counterfactual: identical estimator with season-S evidence removed (posterior - prior)"
    out = []
    for p in players:
        pid, gid = p["player_id"], p["game_id"]
        for q in OPPORTUNITY_QUANTITIES + EFFICIENCY_QUANTITIES:
            r = p.get(q)
            if r is None:
                continue
            opp = q in OPPORTUNITY_QUANTITIES
            f = _base(f"{MODEL_VERSION}:{pid}:{gid}:{q}", "opportunity" if opp else "efficiency",
                      pid, "player", gid, as_of, src_id, verified)
            val = r.get("posterior_conserved", r["posterior"])
            f.update(observation=r["current_estimate"], value=val, unit=UNITS[q], feature_name=q,
                     numerical_effect=val - r["prior_estimate"], numerical_effect_unit=UNITS[q],
                     numerical_effect_method=cf, support_games=r["current_games"],
                     support_opportunities=r["current_denominator"],
                     uncertainty=f"posterior sd {r['posterior_sd']:.4g}; regime {r['regime']}",
                     rationale=(f"{r['prior_source']} {r['prior_estimate']:.4g} "
                                f"(prior games {r['prior_games']}) blended with current "
                                f"{'none' if r['current_estimate'] is None else format(r['current_estimate'], '.4g')}"
                                f" over {r['current_games']} games; w_current {r['w_current']:.3f}, k {r['k']:.4g}"))
            if opp and applied:
                f.update(status="numeric_applied", consumed=True)
            elif not opp:
                f["reason_not_applied"] = ("opportunity-first challenger keeps the incumbent efficiency; "
                                           "efficiency posterior reported for comparison only")
            else:
                f["reason_not_applied"] = "shadow consumption"
            out.append(f)
        for q, (vol, name) in SHARE_VOLUME.items():
            if p.get(name) is None:
                continue
            f = _base(f"{MODEL_VERSION}:{pid}:{gid}:{name}", "opportunity", pid, "player", gid,
                      as_of, src_id, verified)
            f.update(value=p[name], unit=UNITS[name], feature_name=name,
                     numerical_effect=p[name] - p[f"{name}_prior_only"], numerical_effect_unit=UNITS[name],
                     numerical_effect_method="counterfactual: team volume x share with season-S evidence removed",
                     support_games=p[q]["current_games"], support_opportunities=p[q]["current_denominator"],
                     rationale=f"team {vol} posterior x conserved {q}",
                     reason_not_applied=None if applied else "shadow consumption")
            if applied:
                f.update(status="numeric_applied", consumed=True)
            out.append(f)
        for name, why in (("routes_run", "routes are not in the supplied data; targets are not a route proxy"),
                          ("snaps", "snap counts are not in the supplied data")):
            f = _base(f"{MODEL_VERSION}:{pid}:{gid}:{name}", "participation", pid, "player", gid,
                      as_of, src_id, False)
            f.update(measurement_kind="unavailable", status="unavailable_unverified", feature_name=name,
                     reason_not_applied=why, rationale=why, cutoff_ok=None)
            out.append(f)
        f = _base(f"{MODEL_VERSION}:{pid}:{gid}:role_regime", "regime", pid, "player", gid, as_of,
                  src_id, verified)
        f.update(measurement_kind="observed", status="context_only", feature_name="role_regime",
                 value=p["regime"], observation=p["prior_season_team"],
                 reason_not_applied="regime selects the learned prior strength; it adds no numeric boost",
                 rationale=f"prior-season team {p['prior_season_team']} vs current {p['team']}")
        out.append(f)
    for ev in regime_events or ():
        f = _base(f"{MODEL_VERSION}:{ev.get('player_id')}:event:{ev.get('event')}", "regime",
                  ev.get("player_id"), "player", ev.get("game_id"), as_of,
                  ev.get("source_id", src_id), bool(ev.get("verified", False)))
        f.update(measurement_kind="observed", status="context_only", feature_name="regime_event",
                 value=ev.get("event"), published_at=ev.get("published_at"),
                 reason_not_applied="no tested numeric mapping for news regime events",
                 rationale=str(ev.get("event")))
        out.append(f)
    for team, tv in teams.items():
        for q in TEAM_QUANTITIES:
            r = tv[q]
            f = _base(f"{MODEL_VERSION}:{team}:{q}", "opportunity", team, "team", None, as_of, src_id, verified)
            f.update(value=r["posterior"], observation=r["current_estimate"], unit=UNITS[q], feature_name=q,
                     numerical_effect=r["posterior"] - r["prior_estimate"], numerical_effect_unit=UNITS[q],
                     numerical_effect_method=cf, support_games=r["current_games"],
                     support_opportunities=r["current_denominator"],
                     uncertainty=f"posterior sd {r['posterior_sd']:.4g}",
                     rationale=f"prior {r['prior_estimate']:.4g}, w_current {r['w_current']:.3f}",
                     reason_not_applied=None if applied else "shadow consumption")
            if applied:
                f.update(status="numeric_applied", consumed=True)
            out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Adapter (pure): incumbent player_week -> games schema
# --------------------------------------------------------------------------- #
def player_games_from_player_week(pw: pd.DataFrame) -> pd.DataFrame:
    """Map a built ``features.build_player_week`` frame onto the ``games`` schema.

    ``game_id`` is derived as season_week_<sorted team pair>, matching
    ``analysis/football_only_eval.game_id_col``.  Returns a new frame.
    """
    cols = {"rec_yards": "rec_yards", "rush_yards": "rush_yards", "pass_yards": "pass_yards"}
    df = pw.rename(columns={"role": "position"})[
        ["season", "week", "player_id", "team", "defteam", "position"] + STAT_COLS + TEAM_COLS
    ].rename(columns=cols).copy()
    pair = ["_".join(sorted((str(a), str(b)))) for a, b in zip(df["team"], df["defteam"])]
    df["game_id"] = (df["season"].astype(int).astype(str) + "_"
                     + df["week"].astype(int).map("{:02d}".format) + "_" + pd.Series(pair, index=df.index))
    return df.drop(columns=["defteam"])
