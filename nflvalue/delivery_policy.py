"""Pre-send authorization for a pick, and the approval line the user reads.

Authorizing a send and recording what was sent are different acts. This module
decides the first; ``issued_ledger.record_delivered`` always performs the second
and only *flags* what this module would have blocked.

    approved   the card's status is ``actionable`` (today: a market that passed the
               offered-line calibration gate -- ``pick_cards.VALIDATED_MARKETS``)
    exception  not approved, but an EXPLICIT, complete exception is attached:
               who (``by``), why (``reason``), when (``clock``, zoned ISO). An
               exception never changes the approval status: the forecast stays
               unvalidated and the delivered text says so.
    blocked    everything else, including game lines (spread / total / moneyline:
               ledger decision 2026-09-22, do not bet) and incomplete exceptions.
    watch      a ``watch`` card delivered AS a watch item: not a recommendation,
               not a wager; reported for completeness, never authorizes a bet.

Nothing here defaults to an exception, relabels a forecast as approved, or reads
a sportsbook price.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

GAME_LINE_MARKETS = frozenset({"spread", "total", "moneyline", "spread_line", "total_line"})
GAME_LINE_DECISION = "game lines: ledger decision 2026-09-22 do-not-bet"
DECISIONS = ("approved", "exception", "blocked", "watch")
EXCEPTION_FIELDS = ("by", "reason", "clock")


def _zoned(s) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else None


def is_game_line(card: Dict) -> bool:
    return str(card.get("market") or "").lower() in GAME_LINE_MARKETS or card.get("kind") == "game_line"


def exception_problems(exception: Optional[Dict], now: Optional[dt.datetime] = None) -> list:
    """Why an exception is not usable; empty when complete."""
    if exception is None:
        return ["no explicit exception"]
    if not isinstance(exception, dict):
        return ["exception is not a mapping"]
    missing = [k for k in EXCEPTION_FIELDS if not str(exception.get(k) or "").strip()]
    out = [f"exception incomplete: missing {', '.join(missing)}"] if missing else []
    if not missing:
        clock = _zoned(exception.get("clock"))
        if clock is None:
            out.append("exception incomplete: clock is not a zoned ISO time")
        elif now is not None and clock > now:
            out.append("exception clock is in the future")
    return out


def authorize(card: Dict, exception: Optional[Dict] = None, now: Optional[dt.datetime] = None,
              pick_class: str = "recommendation") -> Dict:
    """Decide whether this card may be SENT as ``pick_class``; never changes the card."""
    status = card.get("status")
    game_line = is_game_line(card)
    unvalidated = (f"none — forecast unvalidated at offered lines "
                   f"(model_p_status={card.get('model_p_status') or 'unknown'})")
    reasons: list = []
    if pick_class == "watch":
        if status == "watch" and not game_line:
            decision = "watch"
            reasons.append("watch item: not a recommendation, not a wager")
        else:
            decision = "blocked"
            reasons.append(GAME_LINE_DECISION if game_line else f"card status {status!r} is not a watch item")
        return {"decision": decision, "reasons": reasons, "approval_status": unvalidated, "game_line": game_line,
                "card_status": status, "exception": None, "exception_complete": False}
    if status == "actionable" and not game_line:
        return {"decision": "approved", "reasons": ["card status actionable: market passed the offered-line gate"],
                "approval_status": "approved: card status actionable (validated market)", "game_line": False,
                "card_status": status, "exception": None, "exception_complete": False}
    if game_line:
        reasons.append(GAME_LINE_DECISION)
    else:
        why = "; ".join(card.get("status_reasons") or []) or "no reason recorded"
        reasons.append(f"card status {status!r} is not actionable ({why})")
    problems = exception_problems(exception, now)
    if problems:
        reasons.extend(problems)
        decision = "blocked"
    else:
        decision = "exception"
        reasons.append(f"explicit exception by {exception['by']}: {exception['reason']}")
    return {"decision": decision, "reasons": reasons, "approval_status": unvalidated, "game_line": game_line,
            "card_status": status, "exception": exception if isinstance(exception, dict) else None,
            "exception_complete": not problems}


def approval_line(result: Dict, risk_cap_units: Optional[float] = None) -> str:
    """One line the delivered text starts with. Never says 'approved' unless it is."""
    exc = result.get("exception") if result.get("exception_complete") else None
    exc_txt = f"by {exc['by']}: {exc['reason']} ({exc['clock']})" if exc else "none"
    cap = f"{float(risk_cap_units):g}u" if isinstance(risk_cap_units, (int, float)) else "none"
    return f"APPROVAL: {result['approval_status']} · EXCEPTION: {exc_txt} · RISK CAP: {cap}"
