"""Phase 8.3 -- plain-English rendering of the contribution ledger.

Two rules make this layer safe:

1. **Every sentence traces to a ledger entry.** Each rendered sentence carries
   the ``entry`` key it came from. :func:`render_case` raises if a sentence is
   produced without one, so prose cannot drift away from the numbers it claims
   to describe.

2. **Numbers are interpolated, never computed.** The renderer receives values
   already formatted by :func:`fmt` from the ledger and splices those exact
   strings in. It does not round, re-derive, unit-convert, or restate. That is
   what makes "every number in the prose matches the ledger" a string-identity
   check rather than a float comparison -- and it is the same contract
   ``synthesis.py`` enforces on the LLM.

Vocabulary
----------
The banned list is not squeamishness. This tool has not proven edge, so
imperative betting language would assert something the evidence does not
support. The permitted register is "lean", "the case is", "evidence is
thin/moderate/strong at n=...". :func:`check_vocabulary` runs over GENERATED
copy (not the templates), because a data value -- a player nicknamed "Lock",
a team note -- can smuggle a banned word into otherwise clean prose.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

#: Imperative betting language. Matched case-insensitively on word boundaries.
BANNED_TERMS = (
    "lock", "locks", "hammer", "hammering", "max bet", "max play",
    "overwhelming", "can't lose", "cannot lose", "free money", "guaranteed",
    "sure thing", "smash", "pound", "fade hard", "bet the", "must bet",
    "no brainer", "no-brainer", "slam", "mortal lock", "best bet of",
    "unmissable", "easy money", "print money", "auto-bet", "load up",
)

#: Words that describe strength and must always be paired with n.
_STRENGTH_WORDS = ("strong", "moderate", "thin", "unproven")


class VocabularyViolation(RuntimeError):
    """Generated copy contained imperative betting language."""


class UntracedSentence(RuntimeError):
    """A sentence was produced without a ledger entry behind it."""


def check_vocabulary(text: str) -> None:
    """Raise if generated copy contains banned imperative betting language."""
    lowered = text.lower()
    hits = [t for t in BANNED_TERMS
            if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", lowered)]
    if hits:
        raise VocabularyViolation(
            f"generated copy contains imperative betting language: {hits}. "
            f"The permitted register is 'lean' / 'the case is' / 'evidence is "
            f"thin|moderate|strong at n=...'.")


def fmt(value, dp: int = 1, suffix: str = "") -> str:
    """The ONLY place a ledger number becomes display text.

    Every rendered figure passes through here, so the prose and the card show
    byte-identical strings and the acceptance test can compare them literally.
    """
    if value is None:
        return "not available"
    return f"{float(value):.{dp}f}{suffix}"


def pct(value, dp: int = 0) -> str:
    if value is None:
        return "not available"
    return f"{float(value) * 100:.{dp}f}%"


# --------------------------------------------------------------------------- #
# Sentences
# --------------------------------------------------------------------------- #
def _sentence(entry: str, text: str) -> Dict:
    if not entry:
        raise UntracedSentence(f"sentence without a ledger entry: {text!r}")
    return {"entry": entry, "text": text}


def render_drivers(ledger: Dict) -> List[Dict]:
    """One traced sentence per contribution."""
    out: List[Dict] = []
    market_unit = ""
    for c in ledger.get("contributions", []):
        key, label = c["key"], c["label"]
        unit = c.get("unit", "")
        market_unit = unit or market_unit
        mult, after = c.get("multiplier"), c.get("value_after")

        if c["direction"] == "baseline":
            out.append(_sentence(key,
                f"Starting point: {fmt(after, 1)} {unit}."))
            continue

        if c["kind"] == "level":
            ref = c.get("reference")
            if ref is None:
                out.append(_sentence(key,
                    f"{label}: {fmt(mult, 3)}, which converts that into "
                    f"{fmt(after, 1)} {unit}."))
            elif c["direction"] == "neutral":
                # Inside the reference's own interval: indistinguishable from
                # average. Saying "below average" here would turn a 0.3% gap
                # into an argument, which is the over-claiming this layer is
                # supposed to prevent.
                n = (c.get("inputs") or {}).get("reference_n")
                out.append(_sentence(key,
                    f"{label}: {fmt(mult, 3)}, statistically indistinguishable "
                    f"from the {fmt(ref, 3)} position average"
                    + (f" (n={n:,})" if n else "")
                    + f" — no argument either way. Gives {fmt(after, 1)} {unit}."))
            else:
                rel = "below" if mult < ref else "above"
                n = (c.get("inputs") or {}).get("reference_n")
                out.append(_sentence(key,
                    f"{label}: {fmt(mult, 3)}, {rel} the {fmt(ref, 3)} "
                    f"position average"
                    + (f" (n={n:,})" if n else "")
                    + f" — that argues {'under' if rel == 'below' else 'over'}. "
                      f"Gives {fmt(after, 1)} {unit}."))
            continue

        if c["direction"] == "neutral":
            out.append(_sentence(key, f"{label}: no effect on this pick."))
            continue

        delta = c.get("delta")
        way = "adds" if (delta or 0) > 0 else "removes"
        out.append(_sentence(key,
            f"{label}: ×{fmt(mult, 3)}, which {way} "
            f"{fmt(abs(delta or 0), 1)} {unit} and gives {fmt(after, 1)} {unit}."))
    return out


def render_evidence(ledger: Dict) -> List[Dict]:
    """One traced sentence per driver's empirical support.

    Never states a strength word without its n, and states plainly when an
    interval was not recomputed rather than leaving a confident-looking gap.
    """
    out: List[Dict] = []
    for c in ledger.get("contributions", []):
        e = c.get("evidence")
        if not e:
            continue
        grade, n = e["grade"], e.get("n")
        ci, status = e.get("interval"), e.get("interval_status")

        if status == "not_applicable":
            out.append(_sentence(c["key"],
                f"{c['label']}: observed from this player's own prior weeks, "
                f"not a claimed effect size."))
            continue

        n_txt = f"n={n:,}" if n else "n not published"
        if ci:
            ci_txt = f"95% CI {fmt(ci[0], 3)}–{fmt(ci[1], 3)}"
        elif status == "not_recomputed":
            ci_txt = "confidence interval not recomputed in this phase"
        else:
            ci_txt = "no interval available"
        out.append(_sentence(c["key"],
            f"{c['label']}: evidence is {grade} at {n_txt}, {ci_txt}. "
            f"{e.get('note') or ''}".strip()))
    return out


def render_counter_case(ledger: Dict, side: Optional[str]) -> List[Dict]:
    """The disconfirming half. Never omitted, even when empty."""
    against = "down" if side == "over" else "up"
    opposing = [c for c in ledger.get("contributions", [])
                if c.get("direction") == against]
    if not opposing:
        return [_sentence("counter_case",
            "Nothing in the projection argues against this side — which is "
            "itself worth noting: a one-sided case usually means few factors "
            "were measured, not that the pick is safe.")]
    out = []
    for c in opposing:
        out.append(_sentence(c["key"],
            f"Against this side: {c['label'].lower()} "
            f"({fmt(c.get('multiplier'), 3)}) pushed the projection "
            f"{'down' if against == 'down' else 'up'} to "
            f"{fmt(c.get('value_after'), 1)} {c.get('unit', '')}."))
    return out


def render_line_note(ledger: Dict, screened: Optional[int] = None,
                     screened_n: Optional[int] = None) -> List[Dict]:
    """Line provenance and the denominator. Both are honesty requirements."""
    out = []
    if ledger.get("is_synthetic_line"):
        out.append(_sentence("line_source",
            f"The {fmt(ledger.get('line'), 1)} reference is SYNTHETIC — the "
            f"player's own trailing mean, not a bookmaker price. No edge can "
            f"be computed against it, and none is shown."))
    elif ledger.get("line_source") == "real":
        out.append(_sentence("line_source",
            f"Line {fmt(ledger.get('line'), 1)} is a real bookmaker price."))
    else:
        out.append(_sentence("line_source", "No line is attached to this pick."))

    if screened_n:
        out.append(_sentence("screen_count",
            f"This surfaced as {screened} of {screened_n} candidates screened "
            f"in this game — the more that are screened, the more the top of "
            f"the list contains luck."))
    return out


def render_what_would_change(ledger: Dict) -> Dict:
    """One line naming the input most able to flip the case."""
    tilts = [c for c in ledger.get("contributions", [])
             if c.get("kind") == "tilt" and c.get("direction") in ("up", "down")]
    if tilts:
        biggest = max(tilts, key=lambda c: abs((c.get("multiplier") or 1) - 1))
        return _sentence(biggest["key"],
            f"What would change this: {biggest['label'].lower()} is doing the "
            f"most work here (×{fmt(biggest.get('multiplier'), 3)}). If that "
            f"moves back to neutral, the case largely goes with it.")
    levels = [c for c in ledger.get("contributions", []) if c.get("kind") == "level"]
    if levels:
        c = levels[0]
        return _sentence(c["key"],
            f"What would change this: the projection rests on {c['label'].lower()} "
            f"holding at {fmt(c.get('multiplier'), 3)}. A usage or role change "
            f"moves it directly.")
    return _sentence("baseline_volume",
        "What would change this: any change to projected volume moves the "
        "number proportionally.")


def render_case(ledger: Dict, side: Optional[str] = None,
                screened: Optional[int] = None,
                screened_n: Optional[int] = None) -> Dict:
    """Full traced narrative for one pick. Every sentence carries its entry."""
    side = side or ledger.get("side")
    blocks = {
        "line": render_line_note(ledger, screened, screened_n),
        "drivers": render_drivers(ledger),
        "counter_case": render_counter_case(ledger, side),
        "evidence": render_evidence(ledger),
        "what_would_change": [render_what_would_change(ledger)],
    }
    all_sentences = [s for group in blocks.values() for s in group]
    for s in all_sentences:
        if not s.get("entry"):
            raise UntracedSentence(f"untraced sentence: {s}")
    joined = " ".join(s["text"] for s in all_sentences)
    check_vocabulary(joined)
    return {"blocks": blocks, "sentence_count": len(all_sentences),
            "text": joined}


def numbers_in(text: str) -> List[str]:
    """Every numeric token in a string -- used by the acceptance test that
    asserts the prose invents nothing."""
    return re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)
