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
