"""QB readiness: who is expected to start, as of when, and on what evidence.

Why this exists (root cause, 2026 W3 final replay)
--------------------------------------------------
``advanced_features.build_qb_continuity`` keys on the schedule's
``home_qb_id`` / ``away_qb_id``. The 2019-2023 base schedule carries those
columns; ``historical/lines_extra.parquet`` (every 2024+ game) does not. The
builder silently skips a game with no QB id, so ``qb_continuity`` is NaN for
EVERY 2024-2026 row -- in the ranker's training frame (2024-2026: 100% NaN,
2019-2023: 0%) and on every live candidate. Consequences:

* ``candidates.apply_backup_qb_adjustment`` skips all rows (mask needs notna);
* the gbdt ranker routes the NaN down its learned missing-value branch, which
  it learned from the 2024+ era rows -- an era indicator, not a starter
  judgement; no starter value is imputed into ``cands``.

Restoring the column verbatim is NOT restoring the measured stage:

* the 2019-2023 schedule ids match the REALIZED starter; their pregame
  publication lineage is unverified (no source establishes when each id was
  published), so they cannot stand in for what a pregame run knew;
* the trailing window is the last 60 (week, passer) rows, ~3 seasons, so a
  second-year starter reads < 0.5: 44% of 2019-2023 team-weeks (56% in 2023)
  fall under the 0.5 threshold, while the x0.92 was measured on 162 backup
  team-weeks. Feeding it would cut roughly half the pass-family board.

So this module exposes the useful, checkable input -- a PROXY for the last
starter (pbp, strictly before the target week: the first pass attempt by a
roster QB when positions are supplied, else the first passer, labelled
unverified -- a trick-play passer is not a starter) and any sourced intended
starter claim that is explicitly confirmed, linked to a unique official roster
id, published before the decision clock AND captured by the issuing run before
that clock -- as CONTEXT, and keeps the numeric consumers blocked with a
machine-readable cause. A claim that was public before the clock but captured
later is kept as a historical-availability note, never as what the run knew.
It never writes ``qb_continuity`` or ``backup_qb_adj``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Per-team resolution states (machine-readable; stable strings).
# A sourced claim is compared with a PROXY for last week's starter, so neither
# outcome is a verified starter change/continuity.
SOURCED_SAME = "sourced_starter_matches_prior_proxy"     # claim == prior-starter proxy
SOURCED_CHANGED = "sourced_starter_differs_from_prior_proxy"  # claim != prior-starter proxy
UNCONFIRMED = "starter_unconfirmed"            # no usable claim; prior starter only
CONFLICT = "starter_conflict"                  # >1 distinct usable claimed starters
NO_PRIOR = "no_prior_realized_starter"         # usable claim but no realized history
UNKNOWN = "unknown"                            # neither claim nor history
STATES = (SOURCED_SAME, SOURCED_CHANGED, UNCONFIRMED, CONFLICT, NO_PRIOR, UNKNOWN)

# Claim rejection reasons.
REJ_TEAM = "team_mismatch"
REJ_NO_CLOCK = "no_published_clock"
REJ_FUTURE = "published_after_as_of"
REJ_POSTGAME = "published_at_or_after_kickoff"
REJ_IDENTITY_NONE = "identity_no_roster_match"
REJ_IDENTITY_AMBIG = "identity_ambiguous"
REJ_NOT_CONFIRMED = "claim_not_confirmed"          # claim_kind must be exactly "confirmed"
REJ_NO_CAPTURE = "no_capture_clock"               # no record of when this run retrieved it
REJ_CAPTURED_LATE = "captured_after_as_of"        # retrieved after the decision clock

# How the prior-starter proxy was chosen.
PRIOR_BASIS_ROSTER_QB = "first_pass_attempt_by_roster_qb"
PRIOR_BASIS_FIRST_PASSER = "first_passer_position_unverified"

# Why the numeric consumers stay blocked even for a verified row.
BLOCK_SOURCE = "schedule_qb_ids_absent_2024plus"
BLOCK_SEMANTICS = "trained_on_realized_starter_window_unvalidated"
NUMERIC_BLOCK = {
    "qb_continuity": [BLOCK_SOURCE, BLOCK_SEMANTICS],
    "backup_qb_adj": [BLOCK_SOURCE, BLOCK_SEMANTICS],
}

CONTEXT_COLUMNS = ("qb_ready_state", "qb_ready_qb_id", "qb_ready_prior_qb_id",
                   "qb_ready_prior_game_id", "qb_ready_source", "qb_ready_published_at",
                   "qb_ready_as_of", "qb_ready_trailing_share", "qb_ready_numeric_blocked")


def _ts(x) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == "":
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _norm(name) -> str:
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower().replace("-", " "))
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"})


def schedule_qb_coverage(schedules: pd.DataFrame) -> Dict[int, float]:
    """Share of REG games per season carrying both schedule QB ids (0.0 when
    the columns are absent) -- the diagnostic that exposes the 2024+ gap."""
    reg = schedules[schedules["game_type"] == "REG"]
    out = {}
    for s, g in reg.groupby("season"):
        if not {"home_qb_id", "away_qb_id"} <= set(g.columns):
            out[int(s)] = 0.0
        else:
            out[int(s)] = round(float((g["home_qb_id"].notna() & g["away_qb_id"].notna()).mean()), 4)
    return out


def prior_realized_starters(pbp: pd.DataFrame, season: int, week: int,
                            positions: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
    """{team: {qb_id, game_id, season, week, basis}} -- a PROXY for the starter of
    each team's most recent game strictly before (season, week).

    With ``positions`` ({player_id: position}, e.g. the official roster) it is the
    first pass attempt by a player listed as QB (basis ``PRIOR_BASIS_ROSTER_QB``);
    without them it is the first passer, which can be a trick-play WR/RB (basis
    ``PRIOR_BASIS_FIRST_PASSER``). Either way it is a completed-game proxy, not a
    sourced starter designation and never a claim about this week."""
    d = pbp[(pbp["pass_attempt"] == 1) & pbp["passer_player_id"].notna()]
    d = d[(d["season"] < season) | ((d["season"] == season) & (d["week"] < week))]
    basis = PRIOR_BASIS_FIRST_PASSER
    if positions:
        d = d[d["passer_player_id"].map(lambda p: positions.get(p) == "QB")]
        basis = PRIOR_BASIS_ROSTER_QB
    if d.empty:
        return {}
    sort = [c for c in ("season", "week", "game_id", "play_id") if c in d.columns]
    d = d.sort_values(sort)
    first = d.groupby(["posteam", "season", "week"], sort=True).head(1)
    last = first.groupby("posteam").tail(1)
    return {r.posteam: {"qb_id": r.passer_player_id, "game_id": getattr(r, "game_id", None),
                        "season": int(r.season), "week": int(r.week), "basis": basis}
            for r in last.itertuples(index=False)}


def trailing_share(pbp: pd.DataFrame, team: str, qb_id: str, season: int, week: int,
                   window: int = 60) -> Optional[float]:
    """Same arithmetic as ``build_qb_continuity`` (trailing 60 (week, passer)
    rows) for an explicitly supplied starter. Context only."""
    d = pbp[(pbp["pass_attempt"] == 1) & pbp["passer_player_id"].notna() & (pbp["posteam"] == team)]
    d = d[(d["season"] < season) | ((d["season"] == season) & (d["week"] < week))]
    att = (d.groupby(["season", "week", "passer_player_id"]).size().rename("att")
           .reset_index().sort_values(["season", "week"]).tail(window))
    total = float(att["att"].sum())
    return round(float(att.loc[att["passer_player_id"] == qb_id, "att"].sum()) / total, 4) if total else None


def link_claim_identity(claim: Dict, roster: pd.DataFrame) -> Dict:
    """Attach a unique official roster id to a starter claim.

    ``roster``: [player_id, full_name, team, position] for the target
    season/week. A claim that already carries ``qb_id`` must still appear on
    that team's roster as a QB. Name linking requires team + QB + exact
    normalized full name and exactly one hit; nothing else is guessed."""
    team = claim.get("team")
    qbs = roster[(roster["team"] == team) & (roster["position"] == "QB")]
    if claim.get("qb_id"):
        hit = qbs[qbs["player_id"] == claim["qb_id"]]
        return {**claim, "qb_id": claim["qb_id"] if len(hit) == 1 else None,
                "identity": "roster_id" if len(hit) == 1 else REJ_IDENTITY_NONE}
    want = _norm(claim.get("claim_value") or claim.get("name"))
    hits = qbs[qbs["full_name"].map(_norm) == want].drop_duplicates("player_id")
    if len(hits) == 1:
        return {**claim, "qb_id": hits["player_id"].iloc[0], "identity": "roster_name+team+QB"}
    return {**claim, "qb_id": None,
            "identity": REJ_IDENTITY_AMBIG if len(hits) > 1 else REJ_IDENTITY_NONE}


def starter_claims_from_factor_context(ctx: Dict, game_id: Optional[str] = None) -> List[Dict]:
    """Sourced, confirmed ``qb_news``/``starter`` claims from a factor-context
    document (e.g. data/factor_context/2026-w03.json). Team-specific only; a
    missing team is simply absent -- never a league-wide proxy."""
    out = []
    for n in (ctx or {}).get("news") or []:
        if n.get("category") != "qb_news" or n.get("claim_key") != "starter":
            continue
        if game_id and n.get("game_id") != game_id:
            continue
        out.append({"team": n.get("team"), "claim_value": n.get("claim_value"),
                    "claim_kind": n.get("claim_kind"), "source": n.get("source_url") or n.get("attribution"),
                    "source_tier": n.get("source_tier"), "published_at": n.get("published_at"),
                    "fetched_at": n.get("fetched_at"), "game_id": n.get("game_id")})
    return out


def resolve_team(team: str, prior: Optional[Dict], claims: Iterable[Dict], as_of,
                 kickoff=None) -> Dict:
    """One team's readiness record. Claims must already be identity-linked.

    A claim is usable only if it is explicitly ``confirmed``, for this team, published
    before ``as_of`` (and before kickoff), CAPTURED by the issuing run at or before
    ``as_of`` (``fetched_at``), and linked to a unique roster id. A claim that was
    public before ``as_of`` but captured later is rejected with
    ``published_before_as_of=True``: historical availability, not what this run knew."""
    as_of_t, ko = _ts(as_of), _ts(kickoff)
    if as_of_t is None:
        raise ValueError("resolve_team requires an explicit as_of")
    usable, rejected = [], []
    for c in claims:
        pub, got = _ts(c.get("published_at")), _ts(c.get("fetched_at"))
        why = None
        if c.get("team") != team:
            why = REJ_TEAM
        elif c.get("claim_kind") != "confirmed":
            why = REJ_NOT_CONFIRMED
        elif pub is None:
            why = REJ_NO_CLOCK
        elif pub > as_of_t:
            why = REJ_FUTURE
        elif ko is not None and pub >= ko:
            why = REJ_POSTGAME
        elif got is None:
            why = REJ_NO_CAPTURE
        elif got > as_of_t:
            why = REJ_CAPTURED_LATE
        elif not c.get("qb_id"):
            why = c.get("identity") or REJ_IDENTITY_NONE
        if why:
            rejected.append({**c, "rejected": why,
                             "published_before_as_of": bool(pub is not None and pub <= as_of_t)})
        else:
            usable.append(c)
    ids = sorted({c["qb_id"] for c in usable})
    prior_id = (prior or {}).get("qb_id")
    if len(ids) > 1:
        state, qb = CONFLICT, None
    elif len(ids) == 1:
        qb = ids[0]
        state = NO_PRIOR if not prior_id else (SOURCED_SAME if qb == prior_id else SOURCED_CHANGED)
    else:
        qb = None
        state = UNCONFIRMED if prior_id else UNKNOWN
    src = next((c for c in usable if c["qb_id"] == qb), None) if qb else None
    return {"team": team, "state": state, "qb_id": qb, "prior": prior,
            "source": (src or {}).get("source"), "published_at": (src or {}).get("published_at"),
            "as_of": as_of_t.isoformat(), "kickoff": ko.isoformat() if ko is not None else None,
            "usable_claims": usable, "rejected_claims": rejected,
            "numeric_blocked": dict(NUMERIC_BLOCK)}


def context_frame(cands: pd.DataFrame, records: Dict[str, Dict],
                  pbp: Optional[pd.DataFrame] = None, season: Optional[int] = None,
                  week: Optional[int] = None) -> pd.DataFrame:
    """Per-candidate context columns (``CONTEXT_COLUMNS``), indexed like
    ``cands``. Reads only ``team``; never touches mean/sd/p/line/price or
    ``qb_continuity``. Teams without a record are ``unknown``."""
    rows = []
    for team in cands["team"]:
        r = records.get(team) or {}
        qb = r.get("qb_id")
        share = (trailing_share(pbp, team, qb, season, week)
                 if qb and pbp is not None and season is not None else None)
        rows.append({"qb_ready_state": r.get("state", UNKNOWN), "qb_ready_qb_id": qb,
                     "qb_ready_prior_qb_id": (r.get("prior") or {}).get("qb_id"),
                     "qb_ready_prior_game_id": (r.get("prior") or {}).get("game_id"),
                     "qb_ready_source": r.get("source"),
                     "qb_ready_published_at": r.get("published_at"),
                     "qb_ready_as_of": r.get("as_of"), "qb_ready_trailing_share": share,
                     "qb_ready_numeric_blocked": ",".join(NUMERIC_BLOCK["backup_qb_adj"])})
    return pd.DataFrame(rows, index=cands.index, columns=list(CONTEXT_COLUMNS))
