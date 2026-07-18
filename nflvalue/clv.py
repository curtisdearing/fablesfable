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
                "positive_rate": None, "beat_close_rate": None,
                "avg_point_move": None, "window": window}
    tail = df.tail(window)
    return {
        "n": int(len(df)),
        "window": window,
        "rolling_mean": round(float(tail["clv_prob"].mean()), 5),
        "lifetime_mean": round(float(df["clv_prob"].mean()), 5),
        "positive_rate": round(float((df["clv_prob"] > 0).mean()), 4),
        # User-facing alias: this is the explicit share of entries that beat
        # the same-side consensus close requested by the accuracy protocol.
        "beat_close_rate": round(float((df["clv_prob"] > 0).mean()), 4),
        "avg_point_move": round(float(df["point_moved"].mean()), 3),
    }


# --------------------------------------------------------------------------- #
# Opening line (earliest snapshot) -- P0 durable opens/closes record
# --------------------------------------------------------------------------- #
def opening_prob(conn, game_id: str, market: str, player_id: str, side: str,
                 at_or_after_ts: Optional[str] = None) -> Optional[Dict]:
    """Consensus fair probability of ``side`` from the EARLIEST snapshot at or
    after ``at_or_after_ts`` (or the earliest overall). Mirror image of
    ``snapshot_prob`` (latest); together they bracket the open->close move.
    None if no lines."""
    params: List = [game_id, market, player_id]
    ts_clause = ""
    if at_or_after_ts:
        ts_clause = "AND ts >= ?"
        params.append(at_or_after_ts)
    df = dbmod.query_df(conn, f"""
        SELECT * FROM lines
        WHERE game_id=? AND market=? AND player_id=? {ts_clause}
        """, params)
    if df.empty:
        return None
    ts = df["ts"].min()
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
            prob_kind = "raw_implied"
            probs.append(oddsmath.implied_prob(float(over.iloc[0]["price"])))
            points.append(float(over.iloc[0]["point"]))
    if not probs:
        return None
    return {"ts": ts, "prob": sum(probs) / len(probs),
            "point": sum(points) / len(points), "n_books": len(probs),
            "prob_kind": prob_kind}


def log_open_close_for_week(conn, season: int, week: int,
                            kickoffs: Dict[str, str]) -> pd.DataFrame:
    """Persist the opening (earliest snapshot) and closing (latest pre-kickoff)
    consensus line for EVERY (game, market, player, side) that has snapshots in
    ``lines`` -- not just published leans. This is the durable opens/closes
    record P0 calls for: it lets a real-line reliability/CLV backtest be built
    retroactively once enough weeks accrue. Single-snapshot rows store
    open==close (point_moved 0) rather than being dropped, so coverage stays
    visible. Never fabricates a line."""
    keys = dbmod.query_df(conn, "SELECT DISTINCT game_id, market, player_id, side FROM lines")
    rows: List[Dict] = []
    for k in keys.itertuples(index=False):
        kickoff = kickoffs.get(k.game_id)
        op = opening_prob(conn, k.game_id, k.market, k.player_id, k.side)
        cl = snapshot_prob(conn, k.game_id, k.market, k.player_id, k.side,
                           at_or_before_ts=kickoff)
        if op is None and cl is None:
            continue
        if cl is None:
            cl = op
        if op is None:
            op = cl
        rows.append({
            "season": season, "week": week, "game_id": k.game_id,
            "player_id": k.player_id, "market": k.market, "side": k.side,
            "open_ts": op["ts"], "open_point": round(op["point"], 2),
            "open_prob": round(op["prob"], 5), "open_n_books": op["n_books"],
            "close_ts": cl["ts"], "close_point": round(cl["point"], 2),
            "close_prob": round(cl["prob"], 5), "close_n_books": cl["n_books"],
            "prob_kind": op.get("prob_kind", "devig"),
            "point_moved": round(cl["point"] - op["point"], 2),
            "prob_moved": round(cl["prob"] - op["prob"], 5),
        })
    if rows:
        dbmod.upsert(conn, "line_open_close", rows,
                     ["season", "week", "game_id", "player_id", "market", "side"])
    return pd.DataFrame(rows)
