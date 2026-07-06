"""ESPN league news -> per-player news items (free, editorial, laggy — H4).

This is the pipeline's only free news feed. Honesty about what it is: ESPN
editorial headlines lag beat reporters by minutes-to-hours and the market has
usually moved first (PHASE1_HANDSOFF_DESIGN.md H4) — so news here feeds the
CONTEXT layer (synthesis classification -> context panel -> context_ledger),
never a direct number. Text is UNTRUSTED DATA end to end (H7): it flows only
into the keyword classifier and display fields.

Matching is conservative: an article attaches to a player when the athlete
category id or the player's normalized full name appears; no fuzzy guessing.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ._http import get_json
from ..freshness import stamp_now
from .availability import normalize_name

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


def parse_news(raw: Dict) -> List[Dict]:
    """News payload -> [{text, source, timestamp, athlete_ids, headline}]."""
    if not isinstance(raw, dict) or "articles" not in raw:
        raise ValueError("news payload missing 'articles'")
    out = []
    for a in raw.get("articles", []) or []:
        athlete_ids = []
        for c in a.get("categories") or []:
            aid = c.get("athleteId") or (c.get("athlete") or {}).get("id")
            if aid is not None:
                athlete_ids.append(str(aid))
        text = " — ".join(x for x in (a.get("headline"), a.get("description")) if x)
        if not text:
            continue
        out.append({"text": text[:400], "source": "espn_news",
                    "timestamp": a.get("published") or a.get("lastModified") or "",
                    "athlete_ids": athlete_ids, "headline": a.get("headline", "")})
    return out


def fetch_news(limit: int = 50) -> Dict:
    raw = get_json(f"{SITE}/news", params={"limit": limit})
    return {"items": parse_news(raw), "fetched_at": stamp_now()}


def news_by_player(items: List[Dict], players: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Attach articles to gsis player_ids by normalized FULL-name mention.

    ``players``: frame with [player_id, player_name]. Candidate names are
    abbreviated ("A.St. Brown"), so matching keys on last name + first
    initial against words in the text — conservative: a last name alone is
    not enough unless it is unusual (>=7 chars) and unique in the pool.
    """
    keys: Dict[str, List[str]] = {}
    lastname_pool: Dict[str, List[str]] = {}
    for r in players[["player_id", "player_name"]].drop_duplicates().itertuples(index=False):
        norm = normalize_name(r.player_name).replace(".", " ")
        parts = [p for p in norm.replace(".", " ").split() if p]
        if not parts:
            continue
        last = parts[-1]
        first_initial = parts[0][0] if parts[0] else ""
        keys.setdefault(f"{first_initial}|{last}", []).append(r.player_id)
        lastname_pool.setdefault(last, []).append(r.player_id)

    out: Dict[str, List[Dict]] = {}
    for item in items:
        text_norm = normalize_name(item["text"])
        words = text_norm.split()
        matched: set = set()
        for i, w in enumerate(words[1:], start=1):
            key = f"{words[i-1][0]}|{w}"
            for pid in keys.get(key, []):
                matched.add(pid)
        for w in set(words):
            if len(w) >= 7 and len(lastname_pool.get(w, [])) == 1:
                matched.add(lastname_pool[w][0])
        for pid in matched:
            out.setdefault(pid, []).append(
                {"text": item["text"], "source": item["source"], "timestamp": item["timestamp"]})
    return out
