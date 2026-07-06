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
RAW implied probability (vig included) rather than de-vigged. A CLV is only
meaningful when entry and close are the SAME ``prob_kind`` -- subtracting a
raw-implied prob from a de-vigged one is apples-to-oranges. ``snapshot_prob``
tags each snapshot with its ``prob_kind``; ``log_close_for_week`` refuses to
resolve any lean whose entry and close kinds differ, and the resolved ``clv``
row persists the (shared) ``prob_kind`` so nobody mistakes one for the other.

Close is approximate by design (last pre-kickoff snapshot, however old).
``close_ts`` is stored so staleness is always visible.

DIRECTIONALITY RULE: CLV is AFTER-THE-FACT feedback only. It grades whether
past entries beat the market and tunes future selection thresholds
(kill-check, selector.picks_record). It must never flow forward into a live
pick's value -- a line that moved toward our side does NOT make the currently
available number better (if anything, the remaining value is smaller). The
post-projection selector (selector.py) therefore takes no movement/entry-
history inputs at all, and a test pins that invariance.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import pandas as pd

from . import db as dbmod
from . import oddsmath

# Phase 7.3 GAP #1: the closing snapshot must fall within this many hours of
# kickoff -- a snapshot older than that is stale-entry-era, not a close, and
# resolving against it would fake CLV ~= 0. Config "clv.close_window_hours".
DEFAULT_CLOSE_WINDOW_H = 6.0


def _shift_hours(ts: str, hours: float) -> str:
    """ISO8601 UTC timestamp shifted by ``hours`` (may be negative)."""
    t = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    t = t + dt.timedelta(hours=hours)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def has_close_snapshot(conn, game_id: str, kickoff: str,
                       close_window_hours: float = DEFAULT_CLOSE_WINDOW_H) -> bool:
    """True if ``lines`` already has ANY snapshot for ``game_id`` inside the
    closing window ``[kickoff - close_window_hours, kickoff]``.

    Scheduling guard (Phase 7.4): the in-season T-90 job may run several
    times while a game sits inside its close window (cron granularity is
    coarser than the window), and ``resnap_lines`` has no built-in dedup --
    every call spends budget again. Callers should skip games where this
    returns True so at most one close snapshot is paid for per game, per the
    Gap #2 budget reservation (which assumes exactly one close pull/game)."""
    window_start = _shift_hours(kickoff, -close_window_hours)
    n = dbmod.query_df(conn, """
        SELECT COUNT(*) AS n FROM lines
        WHERE game_id=? AND ts >= ? AND ts <= ?
        """, (game_id, window_start, kickoff)).iloc[0]["n"]
    return bool(int(n) > 0)


# --------------------------------------------------------------------------- #
# Snapshot -> consensus de-vigged prob for one (game, market, player, side)
# --------------------------------------------------------------------------- #
def snapshot_prob(conn, game_id: str, market: str, player_id: str, side: str,
                  at_or_before_ts: Optional[str] = None,
                  at_or_after_ts: Optional[str] = None) -> Optional[Dict]:
    """Consensus fair probability of ``side`` from the latest snapshot at or
    before ``at_or_before_ts`` (or the latest overall). None if no lines.

    ``at_or_after_ts`` (Phase 7.3 GAP #1) is a FLOOR -- pass the start of the
    closing window (``kickoff - CLOSE_WINDOW_H``) to require a snapshot that
    actually falls in that window; without it, a snapshot from days earlier
    would silently pass as a "close" and fake CLV ~= 0."""
    params: List = [game_id, market, player_id]
    ts_clause = ""
    if at_or_before_ts:
        ts_clause += " AND ts <= ?"
        params.append(at_or_before_ts)
    if at_or_after_ts:
        ts_clause += " AND ts >= ?"
        params.append(at_or_after_ts)
    df = dbmod.query_df(conn, f"""
        SELECT * FROM lines
        WHERE game_id=? AND market=? AND player_id=? {ts_clause}
        """, params)
    if df.empty:
        return None
    ts = df["ts"].max()
    snap = df[df["ts"] == ts]

    # over/under are paired ONLY within the same (book, point) -- alt lines at
    # a book must never de-vig against each other. When a book quotes several
    # two-sided points, the MAJORITY point across books wins (deterministic
    # tie-break toward the lowest point), and probs average at that point only.
    pairs: List[tuple] = []                 # (book, point, over_price, under_price)
    yes_rows: List[tuple] = []              # (book, point, over_price)
    for (book, point), grp in snap.groupby(["book", "point"]):
        over = grp[grp["side"] == "over"]
        under = grp[grp["side"] == "under"]
        if not over.empty and not under.empty:
            pairs.append((book, float(point), float(over.iloc[0]["price"]),
                          float(under.iloc[0]["price"])))
        elif market == "anytime_td" and not over.empty and side == "over":
            yes_rows.append((book, float(point), float(over.iloc[0]["price"])))

    probs, points, prob_kind = [], [], "devig"
    if pairs:
        counts: Dict[float, int] = {}
        for _, pt, _, _ in pairs:
            counts[pt] = counts.get(pt, 0) + 1
        main_pt = sorted(counts, key=lambda p: (-counts[p], p))[0]
        for _, pt, over_p, under_p in pairs:
            if pt != main_pt:
                continue
            po, pu = oddsmath.devig_multiplicative([over_p, under_p])
            probs.append(po if side == "over" else pu)
            points.append(pt)
    elif yes_rows:
        prob_kind = "raw_implied"           # one-sided market: vig NOT removed
        for _, pt, price in yes_rows:
            probs.append(oddsmath.implied_prob(price))
            points.append(pt)
    if not probs:
        return None
    return {"ts": ts, "prob": sum(probs) / len(probs),
            "point": sum(points) / len(points), "n_books": len(probs),
            "prob_kind": prob_kind}


# --------------------------------------------------------------------------- #
# Entry + close logging
# --------------------------------------------------------------------------- #
def log_close_for_week(conn, season: int, week: int,
                       kickoffs: Dict[str, str],
                       close_window_hours: float = DEFAULT_CLOSE_WINDOW_H) -> pd.DataFrame:
    """For every ACTIVE lean of (season, week) with a real (odds_api) line,
    compute entry prob (latest snapshot <= lean.as_of) and close prob (the
    closing-window snapshot, GAP #1), upsert into ``clv``. Returns the
    resolved rows.

    ``kickoffs``: {game_id: iso kickoff ts} (from the schedules table).
    Leans without any line snapshots resolve to nothing -- visibly absent,
    never faked.

    GAP #3 (clock dedup): a ``wed`` and a ``t90`` lean can exist for the same
    (game, player, market, side) -- the ``clv`` table's PK omits ``clock``, so
    both would collide on upsert. Resolved against the EARLIEST active
    ``as_of`` per key (the wed entry captures the most line movement --
    premortem F5) so exactly one candidate row is computed per key.
    """
    leans = dbmod.query_df(conn, """
        SELECT * FROM leans
        WHERE season=? AND week=? AND status='active' AND line_source='odds_api'
        """, (season, week))
    if leans.empty:
        return pd.DataFrame()
    leans = leans.sort_values("as_of").drop_duplicates(
        subset=["game_id", "player_id", "market", "side"], keep="first")

    rows: List[Dict] = []
    for l in leans.itertuples(index=False):
        kickoff = kickoffs.get(l.game_id)
        if not kickoff:
            continue
        entry = snapshot_prob(conn, l.game_id, l.market, l.player_id, l.side,
                              at_or_before_ts=l.as_of)
        window_start = _shift_hours(kickoff, -close_window_hours)
        close = snapshot_prob(conn, l.game_id, l.market, l.player_id, l.side,
                              at_or_before_ts=kickoff, at_or_after_ts=window_start)
        if entry is None or close is None or close["ts"] <= entry["ts"]:
            continue  # need two distinct snapshots (close inside the window) to say anything
        if entry["prob_kind"] != close["prob_kind"]:
            # Mixed prob_kind (e.g. anytime_td de-vigged at entry but one-sided
            # raw-implied at close, or vice-versa): subtracting these is
            # apples-to-oranges and would feed a bogus number to the kill-check.
            # Refuse to resolve -- visibly absent, never faked.
            continue
        rows.append({
            "season": season, "week": week, "game_id": l.game_id,
            "player_id": l.player_id, "market": l.market, "side": l.side,
            "entry_ts": entry["ts"], "entry_point": entry["point"],
            "entry_price": l.price, "entry_prob": round(entry["prob"], 5),
            "close_ts": close["ts"], "close_point": close["point"],
            "close_price": None, "close_prob": round(close["prob"], 5),
            "clv_prob": round(close["prob"] - entry["prob"], 5),
            "point_moved": round(close["point"] - entry["point"], 2),
            "prob_kind": entry["prob_kind"],   # entry == close (checked above)
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
