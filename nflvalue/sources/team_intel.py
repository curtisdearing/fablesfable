"""Collect cited, context-only local NFL team intelligence.

This module intentionally does not scrape article bodies and never changes a
projection.  It reads RSS/Atom metadata (direct feeds when a publisher exposes
one, otherwise Google News RSS as a discovery layer), classifies a small set of
football-relevant signals, and emits a provenance-rich evidence packet for a
human or a downstream verification layer.

All retrieved text is untrusted data.  A local report can corroborate an
availability or role question; only the simulator's structured official feeds
and validated model features may change numbers.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

SCHEMA_VERSION = 1
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
USER_AGENT = (
    "nfl-value-team-intel/1.0 "
    "(context-only NFL research collector; https://github.com/curtisdearing/fablesfable)"
)
VALID_SOURCE_CLASSES = {
    "official_team", "local_outlet", "independent_blog", "reddit",
}

# These labels are deliberately descriptive, not numeric model features.
_CATEGORY_TERMS = {
    "availability": (
        "injur", "dnp", "did not practice", "did not participate",
        "limited practice", "limited at practice", "limited participant",
        "full practice", "full participant",
        "questionable", "doubtful", "ruled out", "inactive", "concussion",
        "illness", "hamstring", "ankle", "knee", "shoulder", "reserve/",
        "injured reserve", "return to practice",
    ),
    "role_usage": (
        "first-team", "first team", "starter", "starting", "benched",
        "depth chart", "reps", "snap", "workload", "rotation", "committee",
        "target share", "carry", "route", "slot", "backfield",
    ),
    "transaction_roster": (
        "signed", "signing", "waived", "released", "promoted", "elevated",
        "activated", "practice squad", "trade", "traded", "claimed",
    ),
    "staff_scheme": (
        "coordinator", "play caller", "play-caller", "calling plays", "scheme",
        "position coach", "head coach", "offensive line combination",
    ),
    "discipline_status": (
        "suspended", "suspension", "disciplinary", "holdout", "hold-in",
        "excused absence", "personal matter", "legal matter",
    ),
    "travel_environment": (
        "travel", "flight", "weather", "wind", "snow", "heat", "altitude",
        "field condition", "international game", "jet lag",
    ),
}

_SIGNAL_QUERY = (
    'practice OR injury OR injured OR "depth chart" OR starter OR reps OR snaps '
    'OR workload OR inactive OR transaction OR suspended OR illness OR travel '
    'OR weather OR coordinator OR "play caller"'
)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


class TeamIntelSchemaError(ValueError):
    """Registry or feed data did not match the expected shape."""


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(value: object) -> Optional[dt.datetime]:
    """Parse RFC-822 or ISO-8601 timestamps into aware UTC datetimes."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        parsed = None
    if parsed is None:
        try:
            parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def clean_text(value: object, limit: int = 400) -> str:
    """Remove feed markup and bound untrusted text before storage/display."""
    text = html.unescape(str(value or ""))
    text = _TAG_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text[:limit]


def classify_signal(text: object) -> List[str]:
    """Return deterministic football-context labels; an empty list is noise."""
    haystack = f" {clean_text(text, limit=4000).lower()} "
    return [label for label, terms in _CATEGORY_TERMS.items() if any(term in haystack for term in terms)]


def _host(url: object) -> str:
    try:
        host = urllib.parse.urlparse(str(url or "")).hostname or ""
    except ValueError:
        return ""
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _domain_matches(host: str, domain: str) -> bool:
    domain = domain.lower().lstrip(".")
    return host == domain or host.endswith("." + domain)


def validate_registry(payload: Mapping) -> Dict:
    """Validate and normalize a version-1 32-team source registry."""
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise TeamIntelSchemaError(f"registry schema_version must be {SCHEMA_VERSION}")
    teams = payload.get("teams")
    if not isinstance(teams, list) or not teams:
        raise TeamIntelSchemaError("registry.teams must be a non-empty list")

    seen_teams = set()
    seen_sources = set()
    normalized = dict(payload)
    normalized["teams"] = []
    for raw_team in teams:
        if not isinstance(raw_team, Mapping):
            raise TeamIntelSchemaError("each registry team must be an object")
        abbr = str(raw_team.get("abbr") or "").strip().upper()
        name = str(raw_team.get("name") or "").strip()
        if not abbr or not name:
            raise TeamIntelSchemaError("every team requires abbr + name")
        if abbr in seen_teams:
            raise TeamIntelSchemaError(f"duplicate team abbreviation: {abbr}")
        seen_teams.add(abbr)

        sources = raw_team.get("sources")
        if not isinstance(sources, list) or not sources:
            raise TeamIntelSchemaError(f"{abbr}: sources must be a non-empty list")
        clean_sources = []
        for raw_source in sources:
            if not isinstance(raw_source, Mapping):
                raise TeamIntelSchemaError(f"{abbr}: every source must be an object")
            source = dict(raw_source)
            source_id = str(source.get("id") or "").strip()
            source_class = str(source.get("source_class") or "").strip()
            domain = str(source.get("domain") or "").strip().lower()
            if not source_id or not source.get("name") or not domain:
                raise TeamIntelSchemaError(f"{abbr}: every source requires id + name + domain")
            if source_class not in VALID_SOURCE_CLASSES:
                raise TeamIntelSchemaError(f"{abbr}/{source_id}: invalid source_class {source_class!r}")
            if source_id in seen_sources:
                raise TeamIntelSchemaError(f"duplicate source id: {source_id}")
            seen_sources.add(source_id)
            source["id"] = source_id
            source["domain"] = domain
            clean_sources.append(source)

        team = dict(raw_team)
        team["abbr"] = abbr
        team["name"] = name
        team["aliases"] = [str(alias).strip() for alias in team.get("aliases", []) if str(alias).strip()]
        team["sources"] = clean_sources
        normalized["teams"].append(team)
    return normalized


def load_registry(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TeamIntelSchemaError("registry root must be an object")
    return validate_registry(payload)


def _team_index(registry: Mapping) -> Dict[str, Dict]:
    return {str(team["abbr"]).upper(): dict(team) for team in registry.get("teams", [])}


def select_teams(registry: Mapping, abbreviations: Sequence[str]) -> List[Dict]:
    index = _team_index(registry)
    wanted = []
    for raw in abbreviations:
        abbr = str(raw).strip().upper()
        if not abbr:
            continue
        if abbr not in index:
            raise TeamIntelSchemaError(f"unknown team abbreviation: {abbr}")
        if abbr not in {team["abbr"] for team in wanted}:
            wanted.append(index[abbr])
    return wanted


def build_google_news_url(team: Mapping) -> str:
    """Build one domain-allowlisted Google News RSS discovery query per team."""
    # dict.fromkeys dedupes while preserving first-seen order, in case two
    # sources for a team ever share a domain.
    domains = list(dict.fromkeys(str(source["domain"]) for source in team.get("sources", [])))
    site_clause = " OR ".join(f"site:{domain}" for domain in domains)
    query = f'"{team["name"]}" ({_SIGNAL_QUERY}) ({site_clause})'
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    return GOOGLE_NEWS_RSS + "?" + urllib.parse.urlencode(params)


def build_requests(registry: Mapping, abbreviations: Sequence[str]) -> List[Dict]:
    """Return direct-feed/social requests plus one discovery request per team."""
    requests: List[Dict] = []
    for team in select_teams(registry, abbreviations):
        for source in team.get("sources", []):
            source_class = str(source.get("source_class") or "")
            if source_class == "reddit" and source.get("feed_url"):
                requests.append({
                    "id": f'{team["abbr"]}:{source["id"]}:reddit',
                    "team": team,
                    "method": "reddit_json",
                    "url": str(source["feed_url"]),
                    "source": source,
                })
            elif source.get("feed_url"):
                requests.append({
                    "id": f'{team["abbr"]}:{source["id"]}:direct',
                    "team": team,
                    "method": "direct_rss",
                    "url": str(source["feed_url"]),
                    "source": source,
                })
        requests.append({
            "id": f'{team["abbr"]}:google_news_rss',
            "team": team,
            "method": "google_news_rss",
            "url": build_google_news_url(team),
            "source": None,
        })
    return requests


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_text(entry: ET.Element, names: Iterable[str]) -> str:
    wanted = set(names)
    for child in entry.iter():
        if _local_name(child.tag) in wanted and child.text:
            return str(child.text)
    return ""


def _entry_link(entry: ET.Element) -> str:
    for child in entry.iter():
        if _local_name(child.tag) != "link":
            continue
        href = child.attrib.get("href")
        if href:
            return str(href).strip()
        if child.text:
            return str(child.text).strip()
    return ""


def _publisher(entry: ET.Element) -> Dict[str, str]:
    for child in entry.iter():
        if _local_name(child.tag) == "source":
            return {"name": clean_text(child.text, 120), "url": str(child.attrib.get("url") or "")}
    return {"name": "", "url": ""}


def _source_for_entry(request: Mapping, publisher_url: str) -> Dict:
    direct = request.get("source")
    if isinstance(direct, Mapping):
        return dict(direct)
    host = _host(publisher_url)
    for source in request["team"].get("sources", []):
        if _domain_matches(host, str(source["domain"])):
            return dict(source)
    return {
        "id": f'{request["team"]["abbr"].lower()}_google_discovery',
        "name": "Google News discovery",
        "domain": host or "news.google.com",
        "source_class": "discovery",
        "url": publisher_url or GOOGLE_NEWS_RSS,
        "access": "discovery metadata only",
    }


def parse_feed(raw: object, request: Mapping, *, fetched_at: object) -> List[Dict]:
    """Parse RSS/Atom bytes into bounded, provenance-rich evidence items."""
    try:
        root = ET.fromstring(raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8"))
    except ET.ParseError as exc:
        raise TeamIntelSchemaError(f'{request.get("id", "feed")}: invalid RSS/Atom XML: {exc}') from exc
    fetched_dt = parse_timestamp(fetched_at)
    if fetched_dt is None:
        raise TeamIntelSchemaError("fetched_at must be a parseable timestamp")

    entries = [node for node in root.iter() if _local_name(node.tag) in {"item", "entry"}]
    items: List[Dict] = []
    for entry in entries:
        title = clean_text(_first_text(entry, ("title",)), 240)
        summary = clean_text(_first_text(entry, ("description", "summary", "content")), 400)
        url = _entry_link(entry)
        if not title or not url:
            continue
        published_raw = _first_text(entry, ("pubDate", "published", "updated", "date"))
        published_dt = parse_timestamp(published_raw)
        timestamp = iso_utc(published_dt or fetched_dt)
        timestamp_basis = "published_at" if published_dt else "fetched_at"
        publisher = _publisher(entry)
        source = _source_for_entry(request, publisher["url"])
        combined = clean_text(f"{title}. {summary}", 640)
        categories = classify_signal(combined)
        fingerprint = "|".join((str(source["id"]), url, title.lower(), timestamp))
        items.append({
            "id": hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20],
            "team": str(request["team"]["abbr"]),
            "team_name": str(request["team"]["name"]),
            "title": title,
            "summary": summary,
            "text": combined,
            "url": url,
            "published_at": iso_utc(published_dt) if published_dt else None,
            "timestamp": timestamp,
            "timestamp_basis": timestamp_basis,
            "fetched_at": iso_utc(fetched_dt),
            "source": f'team_intel:{source["id"]}',
            "source_id": str(source["id"]),
            "source_name": str(source["name"]),
            "source_domain": str(source["domain"]),
            "source_class": str(source["source_class"]),
            "source_registry_match": source["source_class"] != "discovery",
            "publisher_name": publisher["name"] or str(source["name"]),
            "publisher_url": publisher["url"] or str(source.get("url") or ""),
            "collection_method": str(request["method"]),
            "url_kind": "aggregator_redirect" if request["method"] == "google_news_rss" else "publisher_article",
            "discovery_only": request["method"] == "google_news_rss",
            "requires_corroboration": request["method"] == "google_news_rss"
            or source["source_class"] != "official_team",
            "categories": categories,
            "performance_use": "context_only",
        })
    return items


def _social_item(
    *,
    request: Mapping,
    source: Mapping,
    item_id_seed: str,
    title: str,
    text: str,
    url: str,
    published_dt: Optional[dt.datetime],
    fetched_dt: dt.datetime,
    url_kind: str,
    collection_method: str,
) -> Dict:
    """Shared item shape for reddit/X posts: always corroboration-required
    (evidence Tier F -- a lead, never an established fact; see
    docs/TEAM_INTELLIGENCE.md)."""
    timestamp = iso_utc(published_dt or fetched_dt)
    timestamp_basis = "published_at" if published_dt else "fetched_at"
    categories = classify_signal(text)
    fingerprint = "|".join((str(source["id"]), url, item_id_seed, timestamp))
    return {
        "id": hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20],
        "team": str(request["team"]["abbr"]),
        "team_name": str(request["team"]["name"]),
        "title": clean_text(title, 240),
        "summary": clean_text(text, 400),
        "text": clean_text(text, 640),
        "url": url,
        "published_at": iso_utc(published_dt) if published_dt else None,
        "timestamp": timestamp,
        "timestamp_basis": timestamp_basis,
        "fetched_at": iso_utc(fetched_dt),
        "source": f'team_intel:{source["id"]}',
        "source_id": str(source["id"]),
        "source_name": str(source["name"]),
        "source_domain": str(source["domain"]),
        "source_class": str(source["source_class"]),
        "source_registry_match": True,
        "publisher_name": str(source["name"]),
        "publisher_url": str(source.get("url") or ""),
        "collection_method": collection_method,
        "url_kind": url_kind,
        "discovery_only": False,
        "requires_corroboration": True,
        "categories": categories,
        "performance_use": "context_only",
    }


def parse_reddit_json(raw: object, request: Mapping, *, fetched_at: object) -> List[Dict]:
    """Parse Reddit's free public listing JSON (no auth needed) into evidence
    items. Reddit posts are a fan/community source (evidence Tier F): a lead,
    never an established fact, and always requires_corroboration."""
    source = request.get("source") or {}
    fetched_dt = parse_timestamp(fetched_at)
    if fetched_dt is None:
        raise TeamIntelSchemaError("fetched_at must be a parseable timestamp")
    try:
        payload = json.loads(raw if isinstance(raw, (str, bytes, bytearray)) else str(raw))
    except (TypeError, ValueError) as exc:
        raise TeamIntelSchemaError(f'{request.get("id", "reddit")}: invalid JSON: {exc}') from exc
    children = (((payload or {}).get("data") or {}).get("children")) or []
    items: List[Dict] = []
    for child in children:
        data = (child or {}).get("data") or {}
        title = str(data.get("title") or "")
        if not title or data.get("stickied"):
            continue
        body = str(data.get("selftext") or "")
        permalink = str(data.get("permalink") or "")
        url = f"https://www.reddit.com{permalink}" if permalink else str(data.get("url") or "")
        if not url:
            continue
        created = data.get("created_utc")
        published_dt = None
        if created is not None:
            try:
                published_dt = dt.datetime.fromtimestamp(float(created), tz=dt.timezone.utc)
            except (TypeError, ValueError, OSError):
                published_dt = None
        text = f"{title}. {body}".strip(". ")
        items.append(_social_item(
            request=request, source=source, item_id_seed=str(data.get("id") or url),
            title=title, text=text, url=url, published_dt=published_dt, fetched_dt=fetched_dt,
            url_kind="social_post", collection_method="reddit_json",
        ))
    return items


def _parse_response(raw: object, request: Mapping, *, fetched_at: object) -> List[Dict]:
    """Dispatch to the parser matching the request's collection method."""
    method = request.get("method")
    if method == "reddit_json":
        return parse_reddit_json(raw, request, fetched_at=fetched_at)
    return parse_feed(raw, request, fetched_at=fetched_at)


def _fetch_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _dedupe(items: Iterable[Dict]) -> List[Dict]:
    """Prefer direct/official evidence when titles collide within a team."""
    best: Dict[str, Dict] = {}
    ranks = {
        "official_team": 2, "local_outlet": 1, "independent_blog": 1,
        "reddit": -2, "discovery": 0,
    }
    for item in items:
        title_key = re.sub(r"[^a-z0-9]+", " ", item["title"].lower()).strip()
        key = f'{item["team"]}|{title_key}'
        rank = (0 if item["discovery_only"] else 4) + ranks.get(item["source_class"], 0)
        old = best.get(key)
        old_rank = -1 if old is None else (0 if old["discovery_only"] else 4) + ranks.get(old["source_class"], 0)
        if old is None or rank > old_rank or (rank == old_rank and item["timestamp"] > old["timestamp"]):
            best[key] = item
    return sorted(best.values(), key=lambda row: (row["team"], row["timestamp"], row["title"]), reverse=True)


def collect(
    registry: Mapping,
    abbreviations: Sequence[str],
    *,
    hours: float = 96.0,
    include_noise: bool = False,
    timeout: float = 15.0,
    as_of: object = None,
    fetcher: Callable[[str, float], bytes] = _fetch_bytes,
    strict: bool = False,
) -> Dict:
    """Fetch selected teams and return a versioned JSON-serializable packet."""
    registry = validate_registry(registry)
    as_of_dt = parse_timestamp(as_of) or utcnow()
    requests = build_requests(registry, abbreviations)
    items: List[Dict] = []
    source_health: List[Dict] = []
    future_dated = 0
    stale = 0
    unregistered_sources = 0

    for request in requests:
        try:
            raw = fetcher(str(request["url"]), timeout)
            parsed = _parse_response(raw, request, fetched_at=as_of_dt)
        except Exception as exc:  # feed failures are reported; strict mode re-raises
            source_health.append({
                "request_id": request["id"], "team": request["team"]["abbr"],
                "method": request["method"], "url": request["url"],
                "ok": False, "records": 0,
                "error": f"{type(exc).__name__}: {exc}"[:500],
            })
            if strict:
                raise
            continue

        kept = 0
        for item in parsed:
            if not item["source_registry_match"]:
                unregistered_sources += 1
                continue
            item_dt = parse_timestamp(item["timestamp"])
            if item_dt is None:
                continue
            if item_dt > as_of_dt + dt.timedelta(minutes=5):
                future_dated += 1
                continue
            age_hours = (as_of_dt - item_dt).total_seconds() / 3600.0
            if age_hours > hours:
                stale += 1
                continue
            if not item["categories"] and not include_noise:
                continue
            items.append(item)
            kept += 1
        source_health.append({
            "request_id": request["id"], "team": request["team"]["abbr"],
            "method": request["method"], "url": request["url"],
            "ok": True, "records": len(parsed), "kept": kept, "error": None,
        })

    deduped = _dedupe(items)
    selected = [team["abbr"] for team in select_teams(registry, abbreviations)]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": iso_utc(as_of_dt),
        "lookback_hours": float(hours),
        "teams": selected,
        "items": deduped,
        "source_health": source_health,
        "quality": {
            "requests": len(requests),
            "successful_requests": sum(1 for row in source_health if row["ok"]),
            "failed_requests": sum(1 for row in source_health if not row["ok"]),
            "future_dated_items_dropped": future_dated,
            "stale_items_dropped": stale,
            "unregistered_source_items_dropped": unregistered_sources,
            "context_only": True,
        },
        "policy": {
            "article_bodies_scraped": False,
            "projection_mutation_allowed": False,
            "aggregator_items_require_source_verification": True,
            "google_news_use_scope": "local_personal_noncommercial_discovery",
            "public_redistribution_allowed": False,
        },
    }


def render_markdown(packet: Mapping) -> str:
    """Render an agent-readable, linked briefing without adding claims."""
    lines = [
        "# Local Team Intelligence Brief",
        "",
        f'Generated: `{packet.get("generated_at", "unknown")}`  ',
        f'Window: `{packet.get("lookback_hours", "?")} hours`',
        "",
        "> Context only. Links are evidence leads, not numeric projection adjustments. ",
        "> Verify Google News discoveries at the named publisher; official structured ",
        "> injury/inactive feeds remain authoritative for availability gates.",
        "",
    ]
    by_team: Dict[str, List[Mapping]] = {str(team): [] for team in packet.get("teams", [])}
    for item in packet.get("items", []):
        by_team.setdefault(str(item.get("team", "UNK")), []).append(item)
    for team, items in by_team.items():
        lines.extend((f"## {team}", ""))
        if not items:
            lines.extend(("_No qualifying fresh signals collected._", ""))
            continue
        for item in items:
            labels = ", ".join(item.get("categories", [])) or "noise"
            verify = "; verify publisher" if item.get("discovery_only") else ""
            lines.append(
                f'- **{labels}** — [{item.get("title", "untitled")}]({item.get("url", "")}) '
                f'— {item.get("source_name", "unknown source")} '
                f'(`{item.get("timestamp", "unknown")}`{verify})'
            )
        lines.append("")
    failures = [row for row in packet.get("source_health", []) if not row.get("ok")]
    if failures:
        lines.extend(("## Feed failures", ""))
        for row in failures:
            lines.append(f'- `{row.get("request_id")}`: {row.get("error")}')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_packet(
    packet: Mapping,
    json_path: str | Path,
    markdown_path: str | Path | None = None,
) -> None:
    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_path is not None:
        md_target = Path(markdown_path)
        md_target.parent.mkdir(parents=True, exist_ok=True)
        md_target.write_text(render_markdown(packet), encoding="utf-8")


def synthesis_news(packet: Mapping, teams: Optional[Sequence[str]] = None) -> List[Dict]:
    """Return the minimal shape accepted by the existing synthesis news layer."""
    allowed = None if teams is None else {str(team).upper() for team in teams}
    return [
        {"text": item["text"], "source": item["source"], "timestamp": item["timestamp"]}
        for item in packet.get("items", [])
        if allowed is None or str(item.get("team", "")).upper() in allowed
    ]
