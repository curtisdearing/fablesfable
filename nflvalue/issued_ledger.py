"""Immutable ledger of the exact pick cards the system displayed.

The user grades the picks the assistant GIVES; recording them is bookkeeping
done here, automatically, every time cards are written or published. A record
is the displayed card itself (the exact HTML ``pick_cards.render_cards_html``
emits for it, plus the card JSON) with the issuing run's provenance, the
quote identity and the decision clock, as persisted by that run.

Rules:

* Append-only. ``issued_picks`` rows are never updated or deleted (db
  migration 7 installs triggers that abort both). A card whose content changes
  (new line, price, quote, status, text) is a NEW record with ``revision + 1``
  and ``supersedes`` naming the previous record. Re-recording identical content
  is a no-op: the record id is the sha256 of the canonical content.
* Nothing here reconstructs a probability, a quote or a distribution after the
  fact. ``dist`` is written only when the recording process runs the same code
  commit as the issuing run (it is the market's family in that code); otherwise
  it stays NULL and interval metrics are reported as unavailable.
* ``pick_class``: ``recommendation`` (actionable card), ``watch`` (watch-list
  card), ``not_a_pick`` (research/pass: displayed but never executable).
  ``tier`` keeps primary board output, experimental/shadow output and analyst
  overrides apart; graders never pool them.
* Historical candidates reconstructed from replays are never inserted as issued
  picks: only a card object built by ``pick_cards`` (or an explicit analyst
  record carrying its own displayed text) can be recorded.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Dict, Iterable, List, Optional

TIERS = ("primary", "experimental", "analyst_override")
PICK_CLASS = {"actionable": "recommendation", "watch": "watch", "research": "not_a_pick", "pass": "not_a_pick"}
ISSUED_CLASSES = ("recommendation", "watch")
# fields of the card that depend on the rendering clock, not on the decision
_VOLATILE = ("quote_age_hours",)
COLUMNS = ("record_id", "pick_key", "revision", "supersedes", "season", "week", "game_id", "player_id",
           "player_name", "market", "side", "line", "tier", "surface", "card_status", "pick_class",
           "run_id", "code_sha", "forecast_version", "ranker_sha256", "selection_source", "clock",
           "decision_ts", "quote_book", "quote_price", "quote_ts", "model_p_side", "mean", "sd", "dist",
           "display_html", "card_json", "recorded_at")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def pick_key(season: int, week: int, card: Dict, tier: str) -> str:
    return _sha(_canon([int(season), int(week), card.get("game_id"), card.get("player_id"),
                        card.get("market"), tier]))


def _recorded(value) -> Optional[str]:
    return None if value is None or str(value).startswith("unknown") else str(value)


def build_record(card: Dict, season: int, week: int, tier: str = "primary", surface: str = "run_reports",
                 display_html: Optional[str] = None, dist: Optional[str] = None) -> Dict:
    """The content of one ledger record (no revision/chain/recorded_at yet)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choices: {TIERS}")
    if display_html is None:
        from .pick_cards import render_cards_html
        display_html = render_cards_html([card])
    stable = {k: v for k, v in card.items() if k not in _VOLATILE}
    prov = card.get("provenance") or {}
    q = card.get("quote") or {}
    return {
        "season": int(season), "week": int(week), "game_id": card.get("game_id"),
        "player_id": card.get("player_id"), "player_name": card.get("player"),
        "market": card.get("market"), "side": card.get("side"), "line": card.get("line"),
        "tier": tier, "surface": surface, "card_status": card.get("status"),
        "pick_class": PICK_CLASS.get(card.get("status"), "not_a_pick"),
        "run_id": _recorded(prov.get("run_id")), "code_sha": _recorded(prov.get("code_sha")),
        "forecast_version": _recorded(prov.get("forecast_version")),
        "ranker_sha256": prov.get("ranker_sha256"), "selection_source": _recorded(prov.get("selection_source")),
        "clock": card.get("clock"), "decision_ts": card.get("run_as_of"),
        "quote_book": q.get("book"), "quote_price": q.get("price_decimal"), "quote_ts": q.get("captured_at"),
        "model_p_side": card.get("model_p_side"), "mean": card.get("mean"), "sd": card.get("sd"),
        "dist": dist, "display_html": display_html, "card_json": _canon(stable),
    }


# The id covers what was displayed and decided. Surface and dist are excluded so the same
# card published on the site carries the id the run recorded (dist depends on the recorder).
_NOT_CONTENT = ("record_id", "revision", "supersedes", "recorded_at", "surface", "dist")


def _content_id(rec: Dict) -> str:
    return _sha(_canon({k: rec.get(k) for k in COLUMNS if k not in _NOT_CONTENT}))


def _append(conn, rec: Dict, season: int, week: int, card: Dict, tier: str, recorded_at: str) -> bool:
    rec["record_id"] = _content_id(rec)
    if conn.execute("SELECT 1 FROM issued_picks WHERE record_id=?", (rec["record_id"],)).fetchone():
        return False
    rec["pick_key"] = pick_key(season, week, card, tier)
    prev = conn.execute("SELECT record_id, revision FROM issued_picks WHERE pick_key=? "
                        "ORDER BY revision DESC LIMIT 1", (rec["pick_key"],)).fetchone()
    rec["supersedes"], rec["revision"] = (prev[0], prev[1] + 1) if prev else (None, 1)
    rec["recorded_at"] = recorded_at
    conn.execute(f"INSERT INTO issued_picks ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
                 [rec[c] for c in COLUMNS])
    return True


def _dist_for(card: Dict) -> Optional[str]:
    """The market's distribution family, only when this process runs the issuing commit."""
    from .projection import MARKETS
    from .provenance import run_provenance
    code = (card.get("provenance") or {}).get("code_sha")
    if not code or code != run_provenance().get("code_sha"):
        return None
    return (MARKETS.get(card.get("market")) or {}).get("dist")


def record_cards(conn, season: int, week: int, cards: Iterable[Dict], tier: str = "primary",
                 surface: str = "run_reports", recorded_at: Optional[str] = None) -> List[Dict]:
    """Append every card whose displayed content is new; return the records written."""
    recorded_at = recorded_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = []
    for card in cards:
        rec = build_record(card, season, week, tier=tier, surface=surface, dist=_dist_for(card))
        if _append(conn, rec, season, week, card, tier, recorded_at):
            written.append(rec)
    conn.commit()
    return written


def record_analyst_pick(conn, season: int, week: int, card: Dict, display_text: str,
                        recorded_at: Optional[str] = None) -> List[Dict]:
    """An analyst override: the exact text given to the user plus the same card fields.

    ``card`` must carry the quote identity and clocks the pick was given at; nothing is
    filled in from later data."""
    if not display_text or not display_text.strip():
        raise ValueError("an analyst pick needs the exact displayed text")
    rec_card = dict(card)
    rec_card.setdefault("status", "watch")
    recorded_at = recorded_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = build_record(rec_card, season, week, tier="analyst_override", surface="analyst",
                       display_html=display_text)
    wrote = _append(conn, rec, season, week, rec_card, "analyst_override", recorded_at)
    conn.commit()
    return [rec] if wrote else []


def load(conn, season: Optional[int] = None, week: Optional[int] = None) -> List[Dict]:
    sql, args = "SELECT * FROM issued_picks", []
    if season is not None:
        sql += " WHERE season=?" + (" AND week=?" if week is not None else "")
        args = [season] + ([week] if week is not None else [])
    cur = conn.execute(sql + " ORDER BY season, week, pick_key, revision", args)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def export(records: List[Dict]) -> str:
    """Deterministic JSON export (sorted, stable separators) of ledger records."""
    rows = sorted(records, key=lambda r: (r["season"], r["week"], r["pick_key"], r["revision"]))
    return json.dumps({"schema": "fablesfable.issued_picks.v1", "records": rows},
                      sort_keys=True, indent=1, default=str) + "\n"


def publication_records(cards: List[Dict], season: int, week: int, label: str, published_at: str) -> List[Dict]:
    """Ledger-form records for the cards on one published page (``api/hub.json``), no DB write.

    The public site is built from a read-only DB; rebuilding records from the cards it
    displayed yields the ``issued_picks.record_id`` of the same displayed content."""
    out = []
    for card in cards:
        rec = build_record(card, season, week, tier="primary", surface=f"public_site:{label}")
        rec["record_id"] = _content_id(rec)  # same id as the run's record of this displayed card
        rec["pick_key"] = pick_key(season, week, card, "primary")
        rec["revision"], rec["supersedes"], rec["recorded_at"] = None, None, published_at
        out.append(rec)
    return out
