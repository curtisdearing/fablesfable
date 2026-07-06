"""Data-freshness guardrails: the automated substitute for the missing human.

Every external feed the hands-off pipeline consumes (injuries, rosters,
inactives, lines, news, fantasy projections) is registered here as a
:class:`Feed` -- a name, a timestamp, a record count, and a schema flag --
and :func:`gate` turns the set of feeds into a single, explicit decision:

    publish?            -> False the moment a LOAD-BEARING feed is missing,
                           schema-invalid, empty, or older than its staleness
                           threshold (PHASE1_HANDSOFF_DESIGN.md H1/H3/H10:
                           a dead feed must HALT the pipeline, never silently
                           produce confident picks)
    confidence_cap      -> "low" when a non-load-bearing feed is degraded
                           (publish can proceed, but nothing may claim high
                           confidence on top of stale context)
    leakage_suspected   -> True if any feed claims a timestamp AFTER as_of
                           (future-dated data must never inform a projection)

Nothing here fetches anything -- callers stamp feeds at fetch time with
:func:`stamp_now` and hand them in. Pure functions, deterministic, trivially
testable offline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

# Default staleness thresholds (hours), per feed kind. Overridable per-call
# (config.json carries a "freshness" section; see load_thresholds).
DEFAULT_STALENESS_HOURS: Dict[str, float] = {
    "injuries": 36.0,     # ESPN team injuries: should refresh at least daily in-season
    "inactives": 2.0,     # T-90 per-event actives: only meaningful ~90min pre-kick
    "rosters": 26.0 * 7,  # weekly rosters: a week-old snapshot is normal
    "lines": 24.0,
    "news": 48.0,
    "fantasy": 72.0,      # Sleeper projections update a few times per week
}
FALLBACK_STALENESS_HOURS = 48.0

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp_now() -> str:
    """ISO-8601 UTC timestamp for feeds fetched right now."""
    return utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(ts: Union[str, dt.datetime, None]) -> Optional[dt.datetime]:
    """Parse an ISO-ish timestamp; None/unparseable -> None (treated as missing).

    Accepts 'YYYY-MM-DDTHH:MM:SSZ', with/without seconds or offset, or a
    datetime. Naive datetimes are assumed UTC (every stamp this codebase
    writes is UTC).
    """
    if ts is None:
        return None
    if isinstance(ts, dt.datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
    s = str(ts).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    for fmt in None, "%Y-%m-%dT%H:%M%z", "%Y-%m-%d":
        try:
            if fmt is None:
                parsed = dt.datetime.fromisoformat(s)
            else:
                parsed = dt.datetime.strptime(s, fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


@dataclass
class Feed:
    """One external feed's health at gate time."""

    name: str                                  # e.g. "injuries", "fantasy"
    timestamp: Union[str, dt.datetime, None]   # when the data was FETCHED/UPDATED
    n_records: int = 0
    schema_ok: bool = True
    load_bearing: bool = True                  # False: degrade only, never halt
    detail: str = ""                           # free-text for logs/report

    def age_hours(self, as_of: dt.datetime) -> Optional[float]:
        t = parse_ts(self.timestamp)
        if t is None:
            return None
        return (as_of - t).total_seconds() / 3600.0


def gate(
    feeds: Sequence[Feed],
    as_of: Union[str, dt.datetime, None] = None,
    staleness_hours: Optional[Dict[str, float]] = None,
    min_records: int = 1,
) -> Dict:
    """Evaluate feed health -> one explicit publish/confidence decision.

    Returns::

        {publish: bool, confidence_cap: "high"|"low", stale_feeds: [...],
         missing_feeds: [...], future_dated_feeds: [...],
         leakage_suspected: bool, reasons: [...], as_of: iso}

    Rules (fail-loud; PHASE1_HANDSOFF_DESIGN.md H1/H3/H10):
      * load-bearing feed missing / schema-invalid / < min_records / stale
            -> publish=False
      * non-load-bearing feed degraded -> confidence_cap="low"
      * any feed timestamped AFTER as_of -> leakage_suspected=True and that
        feed is treated as UNUSABLE (missing), because future-dated data can
        neither be trusted nor used.
    """
    as_of_dt = parse_ts(as_of) or utcnow()
    thresholds = dict(DEFAULT_STALENESS_HOURS)
    if staleness_hours:
        thresholds.update(staleness_hours)

    publish = True
    cap = "high"
    stale, missing, future = [], [], []
    reasons: List[str] = []

    for f in feeds:
        limit = thresholds.get(f.name, FALLBACK_STALENESS_HOURS)
        age = f.age_hours(as_of_dt)
        problem = None

        if age is not None and age < 0:
            future.append(f.name)
            problem = f"{f.name}: timestamp {f.timestamp} is AFTER as_of ({as_of_dt.isoformat()}) -- future-dated, unusable"
        elif age is None:
            missing.append(f.name)
            problem = f"{f.name}: no/unparseable timestamp -- treated as missing"
        elif not f.schema_ok:
            missing.append(f.name)
            problem = f"{f.name}: schema validation failed ({f.detail or 'unexpected shape'})"
        elif f.n_records < min_records:
            missing.append(f.name)
            problem = f"{f.name}: only {f.n_records} record(s) -- feed empty or truncated"
        elif age > limit:
            stale.append(f.name)
            problem = f"{f.name}: {age:.1f}h old > {limit:.0f}h staleness threshold"

        if problem:
            reasons.append(problem)
            if f.load_bearing:
                publish = False
            cap = "low"

    return {
        "publish": publish,
        "confidence_cap": cap,
        "stale_feeds": stale,
        "missing_feeds": missing,
        "future_dated_feeds": future,
        "leakage_suspected": bool(future),
        "reasons": reasons,
        "as_of": as_of_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def cap_confidence(confidence: str, cap: str) -> str:
    """Apply a gate's confidence cap to a per-player confidence label."""
    c = CONFIDENCE_ORDER.get(confidence, 0)
    m = CONFIDENCE_ORDER.get(cap, 2)
    inv = {v: k for k, v in CONFIDENCE_ORDER.items()}
    return inv[min(c, m)]
