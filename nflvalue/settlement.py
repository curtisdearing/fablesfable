"""Settlement contract for graded prop decisions: win / loss / push / void / unresolved.

Documented participation and settlement rules (US books, player props):

* Two-sided count/yardage markets settle against the player's final stat.
  ``actual > line`` -> OVER wins; ``actual < line`` -> UNDER wins;
  ``actual == line`` (only possible at an integer line) -> PUSH for BOTH
  sides (stake returned).  A push is not a loss and not a hit.
* ``anytime_td`` is yes-only: one or more touchdowns -> WIN, zero -> LOSS.
* A player with NO stat row is UNRESOLVED.  The play-by-play derived
  player-week table only carries players who touched the ball; it cannot
  distinguish "played, zero involvement" (books settle: 0) from "inactive"
  (books void).  Neither a settled zero nor a void may be inferred from the
  absence of a row -- resolution needs a participation source.
* A lean whose status is not ``active`` was never a live decision -> VOID.
* A non-finite or missing actual with a stat row present -> UNRESOLVED.
* A side that is not ``over``/``under`` (``over``/``yes`` for yes-only markets) -> UNRESOLVED.
* ``pass_attempts`` settles on the OFFICIAL attempts stat, which excludes sacks
  and two-point tries.
  nflverse play-by-play sets ``pass_attempt=1`` on every sack, so the
  player-week ``pass_attempts`` column is sack-inclusive (a model feature, left
  as is). Settlement reads ``OFFICIAL_PASS_ATTEMPTS_COL`` (sack-excluded, from
  ``official_pass_attempts``); a table without it leaves the lean UNRESOLVED.
  Checked row by row against ESPN official final boxes, 2026 week 2: sack-
  excluded counts matched 40/40 passers, sack-inclusive 10/40.

Book-specific rules NOT verified in this repository (quoted, not guessed):
``BOOK_RULES_UNVERIFIED`` below. The grader reports them with every grade.

``hit`` is 1/0 only for WIN/LOSS; None otherwise, so every consumer that
averages ``hit`` (learning reliability, why-report, calibration grade) is
forced to settle on settled non-push outcomes.  Rows written before this
contract carry ``settlement=NULL`` and keep their historical ``hit``; they
are read as-is ("legacy_binary") and are never regraded silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

WIN = "win"
LOSS = "loss"
PUSH = "push"
VOID = "void"
UNRESOLVED = "unresolved"
SETTLED = (WIN, LOSS)
SETTLEMENTS = (WIN, LOSS, PUSH, VOID, UNRESOLVED)
YES_ONLY_MARKETS = ("anytime_td",)
OFFICIAL_PASS_ATTEMPTS_COL = "pass_attempts_official"
BOOK_RULES_UNVERIFIED = (
    "player did not play / inactive: void vs. settled (and whether one snap counts as action) is "
    "book-specific and not verified here; no-row players are graded UNRESOLVED, never void or zero",
    "official stat corrections after the game: whether a book resettles is book-specific and not "
    "verified here; grades name their actuals source and capture clock, and a later capture that "
    "differs is reported as a correction beside the original, never silently replaced",
    "push at an integer line (stake returned) is the documented two-sided convention; a book's "
    "own rule for a given market is not verified here",
    "pass attempts: official attempts exclude sacks and two-point tries; whether a book counts "
    "spikes or uses a different stat provider is not verified here",
)


@dataclass(frozen=True)
class Verdict:
    settlement: str
    hit: Optional[int]
    actual: Optional[float]
    detail: str


def _finite(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def settle(market: str, side: str, line, actual, has_stat_row: bool,
           lean_status: str = "active") -> Verdict:
    """Settle one decision.  Never invents a zero from a missing row."""
    if str(lean_status or "active") != "active":
        return Verdict(VOID, None, _finite(actual), f"lean status {lean_status!r}: never a live decision")
    if not has_stat_row:
        return Verdict(UNRESOLVED, None, None, "no stat row: played-with-zero vs inactive is not knowable here")
    a = _finite(actual)
    if a is None:
        return Verdict(UNRESOLVED, None, None, "actual missing or non-finite")
    valid_sides = ("over", "yes") if market in YES_ONLY_MARKETS else ("over", "under")
    if side not in valid_sides:
        return Verdict(UNRESOLVED, None, a, f"invalid side {side!r} for {market}")
    if market in YES_ONLY_MARKETS:
        won = a >= 1.0
        detail = "yes-only market: scored" if won else "yes-only market: did not score"
        return Verdict(WIN if won else LOSS, int(won), a, detail)
    ln = _finite(line)
    if ln is None:
        return Verdict(UNRESOLVED, None, a, "line missing or non-finite")
    if a == ln:
        return Verdict(PUSH, None, a, f"actual {a:g} equals the line {ln:g}: stake returned to both sides")
    over_won = a > ln
    won = over_won if side == "over" else (not over_won)
    detail = "actual landed on the projected side" if won else "actual landed on the other side"
    return Verdict(WIN if won else LOSS, int(won), a, detail)



def official_pass_attempts(pbp):
    """Per (season, week, passer) official pass attempts, with explicit coverage.

    Official attempts = nflverse ``pass_attempt`` plays minus sacks minus two-point tries.
    Two-point tries are identified by ``two_point_attempt == 1`` when that column exists,
    otherwise by a missing ``down`` on the play (nflverse leaves ``down`` empty on
    conversion tries; 2026 wk1 has 4 such pass plays). Without ``sack`` or a way to find
    two-point tries this refuses rather than return a non-official count.

    Only passers who appear on a pass play (including sacks and two-point tries) in the
    given play-by-play are COVERED; a covered passer with no official attempt gets 0.
    Anyone else is absent from the result -- coverage is not inferred."""
    import pandas as pd
    if "sack" not in pbp.columns:
        raise ValueError("play-by-play lacks 'sack': official pass attempts cannot be derived")
    if "two_point_attempt" not in pbp.columns and "down" not in pbp.columns:
        raise ValueError("play-by-play lacks 'two_point_attempt' and 'down': two-point tries "
                         "cannot be excluded")
    plays = pbp[pbp["pass_attempt"] == 1].dropna(subset=["passer_player_id"])
    two_pt = (plays["two_point_attempt"].fillna(0) == 1 if "two_point_attempt" in plays.columns
              else plays["down"].isna())
    counted = (plays["sack"].fillna(0) != 1) & ~two_pt
    out = (plays.assign(_n=counted.astype(float))
           .groupby(["season", "week", "passer_player_id"])["_n"].sum()
           .rename(OFFICIAL_PASS_ATTEMPTS_COL).reset_index()
           .rename(columns={"passer_player_id": "player_id"}))
    return out if len(out) else pd.DataFrame(columns=["season", "week", "player_id", OFFICIAL_PASS_ATTEMPTS_COL])


def with_official_pass_attempts(pw, pbp):
    """``pw`` plus the official attempts column: covered passers get their count (0 is a
    verified 0); everyone else NaN, which settles as UNRESOLVED -- never an invented 0."""
    off = official_pass_attempts(pbp)
    return pw.drop(columns=[OFFICIAL_PASS_ATTEMPTS_COL], errors="ignore").merge(
        off, on=["season", "week", "player_id"], how="left")
