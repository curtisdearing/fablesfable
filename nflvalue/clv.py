"""Forward CLV (closing-line-value) log -- the ONLY honest edge test on free data.

PROP_SHORTLISTER_SPEC.md §5: with no historical prop-line data, "does the
model beat the price" can only be measured forward: log every published lean
with the price at entry, log the last snapshot before kickoff as the
approximate CLOSE, and track whether our entries systematically beat the
close. Positive average CLV is the accepted proxy for real edge; a lean
record without CLV is just a story.

Probabilities are compared in DE-VIGGED space (consensus across the books in
the snapshot), so a book fattening its margin can't masquerade as line
movement. ``anytime_td`` is quoted one-sided (Yes only), so its "prob" is the
RAW implied probability -- vig included at both entry and close, so the
DIFFERENCE is still meaningful; rows carry ``prob_kind`` so nobody mistakes
one for the other.

Close is approximate by design (last pre-kickoff snapshot, however old).
``close_ts`` is stored so staleness is always visible.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from . import db as dbmod
from . import oddsmath


# --------------------------------------------------------------------------- #
# Snapshot -> consensus de-vigged prob for one (game, market, player, side)
# --------------------------------------------------------------------------- #
def snapshot_prob(conn, game_id: str, market: str, player_id: str, side: str,
                  at_or_before_ts: Optional[str] = None) -> Optional[Dict]:
    """Consensus fair probability of ``side`` from the latest snapshot at or
    before ``at_or_before_ts`` (or the latest overall). None if no lines."""
    params: List = [game_id, market, player_id]
    ts_clause = ""
    if at_or_before_ts:
        ts_clause = "AND ts <= ?"
        params.append(at_or_before_ts)
    df = dbmod.query_df(conn, f"""
        SELECT * FROM lines
        WHERE game_id=? AND market=? AND player_id=? {ts_clause}
        """, params)
    if df.empty:
        return None
    ts = df["ts"].max()
    snap = df[df["ts"] == ts]

    probs, points, prob_kind = [], [], "devig"
    for book, grp in snap.groupby("book"):
        over = grp[grp["side"] == "over"]
        under = grp[grp["side"] == "under"]
        if not over.empty and not under.empty:
            po, pu = oddsmath.devig_multiplicative(
                [float(over.iloc[0]["price"]), float(under.iloc[0]["price"])])
            probs.append(po if side == "over" else pu)
            points.append(float(over.iloc[0]["point"]))
        elif market == "anytime_td" and not over.empty and side == "over":
            prob_kind = "raw_implied"          # one-sided market: vig NOT removed
            probs.append(oddsmath.implied_prob(float(over.iloc[0]["price"])))
            points.append(float(over.iloc[0]["point"]))
    if not probs:
        return None
    return {"ts": ts, "prob": sum(probs) / len(probs),
            "point": sum(points) / len(points), "n_books": len(probs),
            "prob_kind": prob_kind}


# --------------------------------------------------------------------------- #
# Entry + close logging
# --------------------------------------------------------------------------- #
def log_close_for_week(conn, season: int, week: int,
                       kickoffs: Dict[str, str]) -> pd.DataFrame:
    """For every ACTIVE lean of (season, week) with a real (odds_api) line,
    compute entry prob (latest snapshot <= lean.as_of) and close prob (latest
    snapshot <= kickoff), upsert into ``clv``. Returns the resolved rows.

    ``kickoffs``: {game_id: iso kickoff ts} (from the schedules table).
    Leans without any line snapshots resolve to nothing -- visibly absent,
    never faked.
    """
    leans = dbmod.query_df(conn, """
        SELECT * FROM leans
        WHERE season=? AND week=? AND status='active' AND line_source='odds_api'
        """, (season, week))
    rows: List[Dict] = []
    for l in leans.itertuples(index=False):
        kickoff = kickoffs.get(l.game_id)
        if not kickoff:
            continue
        entry = snapshot_prob(conn, l.game_id, l.market, l.player_id, l.side,
                              at_or_before_ts=l.as_of)
        close = snapshot_prob(conn, l.game_id, l.market, l.player_id, l.side,
                              at_or_before_ts=kickoff)
        if entry is None or close is None or close["ts"] <= entry["ts"]:
            continue  # need two distinct snapshots to say anything
        rows.append({
            "season": season, "week": week, "game_id": l.game_id,
            "player_id": l.player_id, "market": l.market, "side": l.side,
            "entry_ts": entry["ts"], "entry_point": entry["point"],
            "entry_price": l.price, "entry_prob": round(entry["prob"], 5),
            "close_ts": close["ts"], "close_point": close["point"],
            "close_price": None, "close_prob": round(close["prob"], 5),
            "clv_prob": round(close["prob"] - entry["prob"], 5),
            "point_moved": round(close["point"] - entry["point"], 2),
        })
    if rows:
        dbmod.upsert(conn, "clv", rows,
                     ["season", "week", "game_id", "player_id", "market", "side"])
    return pd.DataFrame(rows)


def rolling_clv(conn, window: int = 50) -> Dict:
    """Mean CLV over the last ``window`` resolved leans (and lifetime)."""
    df = dbmod.query_df(conn, "SELECT * FROM clv ORDER BY close_ts")
    if df.empty:
        return {"n": 0, "rolling_mean": None, "lifetime_mean": None,
                "positive_rate": None, "window": window}
    tail = df.tail(window)
    return {
        "n": int(len(df)),
        "window": window,
        "rolling_mean": round(float(tail["clv_prob"].mean()), 5),
        "lifetime_mean": round(float(df["clv_prob"].mean()), 5),
        "positive_rate": round(float((df["clv_prob"] > 0).mean()), 4),
        "avg_point_move": round(float(df["point_moved"].mean()), 3),
    }
