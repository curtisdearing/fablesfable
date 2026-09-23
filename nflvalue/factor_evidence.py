"""Factor evidence: what was known, where it came from, and whether the model used it.

A card should show material news and context -- QB changes, OL and defensive
absences, role changes, venue/primetime/home-away history -- even when none of
it moves the number, and say truthfully whether the forecast consumed it.  This
module turns plain-dict factor records into that statement.  It computes
nothing about football; it only classifies and renders records supplied by the
pipeline, the news feeds and the opportunity/personnel specialists.

Status is derived from EXECUTION flags, never from what a record claims
-----------------------------------------------------------------------
``normalize_record`` recomputes ``status`` from:

* ``consumed``          the PRIMARY forecast (mean/sd/P) read this input on this run;
* ``consumed_shadow``   only a shadow path read it;
* ``evaluated_neutral`` a stage ran and recorded a neutral result (multiplier 1.0);
* ``populated``         the value was present at run time (False = missing, not zero);
* ``verified`` / ``cutoff_ok`` source and clock checks.

A record that claims ``numeric_applied`` but has ``consumed`` False is
downgraded, and the note says why.  Feature-schema membership, availability, a
join, or a statistical association is not consumption.  The five statuses:

==========================  =====================================================
numeric_applied             primary consumed it.  ``numerical_effect`` is kept only
                            when ``effect_method`` is an isolating method (the
                            executed stage multiplier, a controlled counterfactual,
                            or an ablation).  Otherwise it is None and the label
                            reads "contribution not isolated".
considered_no_change        primary consumed it and recorded a neutral result.
shadow_only                 only a shadow path consumed it; any effect is kept as
                            ``shadow_effect`` and never shown as a projection effect.
context_only                shown for context; the projection does not read it.
unavailable_unverified      missing, stale, unsourced, after the cutoff, or rumor.
==========================  =====================================================

The ML ranker is an ORDERING score, not the forecast.  A feature it reads is
``context_only`` for the projection with ``ordering_consumed=True``, and the
label says it enters the ordering score only.

News is data, never instructions
--------------------------------
Claim text is shown escaped and attributed.  Text that looks like an
instruction or markup is withheld.  News can never promote itself to
``numeric_applied``: only a caller-supplied ``model_links`` entry built from an
executed gate (e.g. the availability OUT gate) can mark it consumed.  Copies
of one story (same ``story_id``) count as one independent source.

Public entrypoints (all pure; inputs are not mutated)
-----------------------------------------------------
normalize_record(raw) -> record
assess_news(items, as_of, model_links=None) -> [record]
records_from_forecast_row(row, receipt, as_of) -> [record]
split_context_record(...) -> record
schedule_context_record(team, games, as_of, sources) -> record
public_label(record) -> label dict
select_for_card(records, card) -> [record]
build_panel(records, as_of) -> panel dict (JSON-serializable)
render_panel_html(panel) -> escaped HTML fragment
"""

from __future__ import annotations

import datetime as dt
import html
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .explain_render import BANNED_TERMS

SCHEMA_VERSION = "factor-evidence-v1"

STATUSES = ("numeric_applied", "considered_no_change", "shadow_only",
            "context_only", "unavailable_unverified")
MEASUREMENT_KINDS = ("observed", "projected", "proxy", "unavailable")

#: Methods that isolate one factor's effect on the published number.
ISOLATING_METHODS = ("executed_stage_multiplier", "controlled_counterfactual", "ablation")

STATUS_LABELS = {
    "numeric_applied": "Used in projection",
    "considered_no_change": "Checked; no change to the projection",
    "shadow_only": "Tested in shadow only; not in the published projection",
    "context_only": "Context only; not used by the projection",
    "unavailable_unverified": "Not verified; not used",
}
ORDERING_ONLY_LABEL = "Enters the ordering score only; does not change the projection"
NOT_ISOLATED = "Used in projection; contribution not isolated"

CATEGORY_LABELS = {
    "availability": "Player availability", "qb_news": "Quarterback news", "team_news": "Team news", "ol_injury": "Offensive line",
    "def_absence": "Defensive absences", "role_usage": "Snaps, touches and targets",
    "matchup": "Matchup", "venue": "Stadium", "home_away": "Home / away",
    "primetime": "Primetime", "schedule": "Schedule", "season_form": "Current season",
    "recent_form": "Recent games", "opponent": "Opponent history", "weather": "Weather",
    "travel": "Travel", "game_script": "Game script", "dispersion": "Spread of outcomes",
    "availability_adjustment": "Injury adjustments", "ordering": "Ordering score inputs",
}
CATEGORY_ORDER = tuple(CATEGORY_LABELS)

#: Source tiers.  Primary = team or league publication.
PRIMARY_TIERS = ("team_official", "league_official")
SOURCE_TIERS = PRIMARY_TIERS + ("data_feed", "media", "third_party_summary", "aggregator")
CLAIM_KINDS = ("confirmed", "report", "rumor")

#: Pre-declared news freshness windows (hours since publication).
MAX_NEWS_AGE_H = {"qb_news": 96, "ol_injury": 96, "def_absence": 96, "role_usage": 168,
                  "team_news": 168}
DEFAULT_MAX_NEWS_AGE_H = 120

#: Descriptive splits with fewer games than this are labelled insufficient.
MIN_SPLIT_GAMES = 5
#: Venue-local kickoff hour at or after which a game counts as primetime.
PRIMETIME_LOCAL_HOUR = 18

_INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above)|disregard|system prompt|you are now|"
    r"<\s*/?\s*(script|iframe|img|a|style)\b|javascript:|\bset status\b|numeric_applied|"
    r"assistant:|tool call|```)", re.I)
_CERTAINTY = ("definitely", "certain to", "will smash", "can't miss", "cannot miss",
              "guarantee", "100% chance", "100% sure", "sure to")
_MAX_CLAIM_CHARS = 280


class FactorRecordError(ValueError):
    """A record violates the shared contract (missing clock, unnamed proxy, ...)."""


class UnsafeCopy(RuntimeError):
    """Generated public copy contains imperative betting or certainty language."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _ts(x) -> Optional[dt.datetime]:
    if x is None or x == "":
        return None
    if isinstance(x, dt.datetime):
        t = x
    else:
        try:
            t = dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        except ValueError:
            return None
    if t.tzinfo is None:
        raise FactorRecordError(f"timestamp {x!r} has no timezone")
    return t


def _iso(t: Optional[dt.datetime]) -> Optional[str]:
    return t.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z") if t else None


def _num(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _as_of(as_of) -> dt.datetime:
    if not isinstance(as_of, (dt.datetime, str)):
        raise FactorRecordError("as_of is required")
    t = _ts(as_of)
    if t is None:
        raise FactorRecordError(f"unparseable as_of {as_of!r}")
    return t


def _is_neutral(effect: float, unit: str) -> bool:
    return abs(effect - (1.0 if unit.startswith("x") else 0.0)) < 1e-9


def _clean_text(text) -> Tuple[Optional[str], bool]:
    """(display text, withheld).  News text is data: never executed or trusted."""
    if text is None:
        return None, False
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(text)).strip()
    if _INJECTION.search(s):
        return None, True
    return (s[:_MAX_CLAIM_CHARS - 1] + "…" if len(s) > _MAX_CLAIM_CHARS else s), False


# --------------------------------------------------------------------------- #
# the core: normalize + derive status
# --------------------------------------------------------------------------- #

_DEFAULTS = dict(
    factor_id=None, category=None, entity_id=None, entity_type=None, game_id=None, team=None,
    as_of=None, observation=None, value=None, unit=None, observed_at=None, published_at=None,
    fetched_at=None, source_url=None, source_id=None, source_title=None, verified=False,
    cutoff_ok=None, measurement_kind="observed", proxy_name=None, status=None, component=None,
    model_version=None, feature_name=None, configured=None, available=None, joined=None,
    populated=True, consumed=False, consumed_shadow=False, ablated=None, evaluated_neutral=False,
    ordering_consumed=False, numerical_effect=None, effect_unit=None, effect_method=None,
    support_games=None, support_opportunities=None, support_scope=None, reason_not_applied=None,
    rationale=None, uncertainty=None, reliability=None, attribution=None, contradiction=False,
    independent_sources=None, copies=None, text_withheld=False, expires_at=None,
)


def normalize_record(raw: Dict) -> Dict:
    """Validate a contract record and derive its status from execution flags.

    Returns a new dict; ``raw`` is not modified.  ``status`` in the input is
    treated as a CLAIM: the derived status replaces it and ``status_notes``
    records any downgrade.
    """
    r = {**_DEFAULTS, **raw}
    for k in ("factor_id", "category"):
        if not r.get(k):
            raise FactorRecordError(f"{k} is required")
    as_of = _as_of(r["as_of"])
    r["as_of"] = _iso(as_of)
    for k in ("observed_at", "published_at", "fetched_at", "expires_at"):
        r[k] = _iso(_ts(r[k]))
    if r["measurement_kind"] not in MEASUREMENT_KINDS:
        raise FactorRecordError(f"measurement_kind {r['measurement_kind']!r}")
    if r["measurement_kind"] == "proxy" and not r.get("proxy_name"):
        raise FactorRecordError("a proxy must be named (proxy_name)")
    if r["measurement_kind"] == "unavailable":
        r["populated"] = False
    if r["cutoff_ok"] is None:
        pub = _ts(r["published_at"]) or _ts(r["observed_at"])
        fetched = _ts(r["fetched_at"])
        r["cutoff_ok"] = not ((pub and pub > as_of) or (fetched and fetched > as_of))

    notes: List[str] = list(raw.get("status_notes") or [])
    pub_t, fetch_t = _ts(r["published_at"]), _ts(r["fetched_at"])
    if pub_t and fetch_t and fetch_t < pub_t:
        notes.append("clock inconsistent: fetched before published")
        r["uncertainty"] = "; ".join(x for x in (r["uncertainty"], "clock inconsistent: "
                                                  "fetched before published") if x)
    claimed = raw.get("status")
    effect, unit = _num(r["numerical_effect"]), r["effect_unit"] or ""
    method = r["effect_method"]
    if effect is not None and method not in ISOLATING_METHODS:
        notes.append(f"effect removed: method {method!r} is not a controlled decomposition "
                      "(a statistical association is not a model effect)")
        effect = None
    shadow_effect = None

    if not r["cutoff_ok"]:
        status = "unavailable_unverified"
        r["reason_not_applied"] = r["reason_not_applied"] or "published or fetched after the decision time"
    elif r["consumed"] and not r["populated"]:
        status = "unavailable_unverified"
        r["reason_not_applied"] = "input missing at run time (missing, not zero)"
    elif r["consumed"]:
        if effect is not None and _is_neutral(effect, unit) and r["evaluated_neutral"]:
            status = "considered_no_change"
        elif effect is None and r["evaluated_neutral"]:
            status = "considered_no_change"
        else:
            status = "numeric_applied"
    elif r["consumed_shadow"]:
        status, shadow_effect, effect = "shadow_only", effect, None
    elif not (r["verified"] or r.get("source_status") == "attributed_report") or not r["populated"]:
        status = "unavailable_unverified"
        r["reason_not_applied"] = r["reason_not_applied"] or (
            "value missing" if not r["populated"] else "not verified")
    else:
        status = "context_only"
    if status not in ("numeric_applied", "considered_no_change"):
        effect = None
    if status == "context_only" and _num(raw.get("numerical_effect")) is not None:
        notes.append("effect removed: the projection did not consume this factor")
    if claimed and claimed != status:
        why = ("presence in a feature schema, availability or a join is not proof of use"
               if claimed in ("numeric_applied", "considered_no_change") and not r["consumed"]
               else "derived from execution flags")
        notes.append(f"claimed {claimed!r} -> {status!r}: {why}")
    if status == "context_only" and r["feature_name"] and not r["ordering_consumed"]:
        if not any("not proof" in n for n in notes):
            notes.append("feature named but not consumed: schema membership is not proof of use")

    r.update(status=status, numerical_effect=effect, shadow_effect=shadow_effect,
             effect_unit=r["effect_unit"] if (effect is not None or shadow_effect is not None) else None,
             status_notes=notes, schema_version=SCHEMA_VERSION)
    if status in ("numeric_applied", "considered_no_change") and not r["reason_not_applied"]:
        r["reason_not_applied"] = None
    elif status == "context_only" and not r["reason_not_applied"]:
        r["reason_not_applied"] = ("enters the ordering score only" if r["ordering_consumed"]
                                   else "not an input to the projection")
    elif status == "shadow_only" and not r["reason_not_applied"]:
        r["reason_not_applied"] = "evaluated in shadow; not promoted to the primary forecast"
    return r


# --------------------------------------------------------------------------- #
# news
# --------------------------------------------------------------------------- #

def _reliability(tier: str, kind: str, attribution: Optional[str]) -> str:
    who = f": {attribution}" if attribution else ""
    if tier in PRIMARY_TIERS and kind == "confirmed":
        return "Confirmed by " + ("league" if tier == "league_official" else "team") + " source"
    if tier == "data_feed" and kind == "confirmed":
        return "Data feed"
    if kind == "rumor":
        return f"Unconfirmed rumor{who}"
    if tier == "third_party_summary":
        return f"Third-party summary (not confirmed){who}"
    return f"Media report (not confirmed){who}"


def assess_news(items: Iterable[Dict], as_of, model_links: Optional[Dict] = None) -> List[Dict]:
    """Source-backed material-news records with cutoff, staleness, dedup, contradiction.

    ``items``: dicts with story_id, entity_id/entity_type, team, game_id, category,
    claim_key (topic, e.g. "availability"), claim_value, claim (short text as
    published), attribution, source_url/source_title/source_id, source_tier,
    claim_kind, published_at/observed_at/fetched_at, optional expires_at.
    ``model_links``: {(entity_id, claim_key): {component, feature_name, consumed,
    reason}} built by the caller from an EXECUTED gate; the only way news counts
    as consumed.  Execution fields inside ``items`` are ignored.
    """
    as_of_t = _as_of(as_of)
    model_links = model_links or {}
    groups: Dict[str, List[Dict]] = {}
    for it in items:
        key = it.get("story_id") or it.get("source_url") or repr(sorted(map(str, it.items())))
        groups.setdefault(str(key), []).append(it)

    rank = {t: i for i, t in enumerate(SOURCE_TIERS)}
    recs: List[Dict] = []
    for story_id, copies in groups.items():
        # the most reliable copy represents the story; copies add no independent support
        it = sorted(copies, key=lambda c: (rank.get(c.get("source_tier"), 99),
                                           str(c.get("published_at") or "")))[0]
        tier, kind = it.get("source_tier") or "aggregator", it.get("claim_kind") or "report"
        text, withheld = _clean_text(it.get("claim"))
        pub, fetched, expires = _ts(it.get("published_at")), _ts(it.get("fetched_at")), _ts(it.get("expires_at"))
        after = bool((pub and pub > as_of_t) or (fetched and fetched > as_of_t))
        reason = None
        if not it.get("source_url"):
            reason = "no public source"
        elif pub is None:
            reason = "publication time unknown"
        elif after:
            reason = "published or fetched after the decision time"
        elif expires and expires <= as_of_t:
            reason = "expired or superseded"
        elif (as_of_t - pub).total_seconds() / 3600 > MAX_NEWS_AGE_H.get(
                it.get("category"), DEFAULT_MAX_NEWS_AGE_H):
            reason = "stale: older than the freshness window"
        elif withheld:
            reason = "claim text withheld: contains instruction-like or markup content"
        elif kind == "rumor":
            reason = "unconfirmed rumor"
        elif tier == "third_party_summary":
            reason = "third-party summary; not confirmed by the primary source"
        verified = reason is None and tier in PRIMARY_TIERS + ("data_feed",) and kind == "confirmed"
        link = model_links.get((it.get("entity_id"), it.get("claim_key"))) or {}
        consumed = bool(link.get("consumed")) and reason is None
        recs.append({
            "factor_id": f"news:{story_id}", "category": it.get("category") or "team_news",
            "entity_id": it.get("entity_id"), "entity_type": it.get("entity_type"),
            "team": it.get("team"), "game_id": it.get("game_id"), "as_of": as_of_t,
            "observation": text, "value": it.get("claim_value"), "unit": it.get("claim_key"),
            "published_at": it.get("published_at"), "observed_at": it.get("observed_at"),
            "fetched_at": it.get("fetched_at"), "expires_at": it.get("expires_at"),
            "source_url": it.get("source_url"), "source_id": it.get("source_id"),
            "source_title": _clean_text(it.get("source_title"))[0],
            "verified": verified, "cutoff_ok": not after,
            "source_status": ("verified" if verified else
                              "attributed_report" if reason is None else "unverified"),
            "populated": bool(it.get("source_url")),
            "measurement_kind": "observed" if it.get("source_url") else "unavailable",
            "component": link.get("component") if consumed else None,
            "feature_name": link.get("feature_name") if consumed else None,
            "consumed": consumed,
            "reason_not_applied": reason or (link.get("reason") if consumed else None),
            "reliability": _reliability(tier, kind, it.get("attribution")),
            "attribution": it.get("attribution"), "text_withheld": withheld,
            "independent_sources": 1, "copies": len(copies),
            "rationale": it.get("rationale"), "uncertainty": it.get("uncertainty"),
            "_topic": (it.get("entity_id"), it.get("claim_key")) if reason is None else None,
            "_val": it.get("claim_value"), "_rank": rank.get(tier, 99),
        })

    # same entity + topic, different values, among in-window records.  A strictly later
    # claim from an equally or more reliable tier SUPERSEDES the earlier ones (status
    # progression, e.g. Monday estimate DNP -> injured reserve); otherwise it is a conflict.
    by_topic: Dict[Tuple, List[Dict]] = {}
    for rec in recs:
        if rec["_topic"] and rec["_topic"][1] and rec["_val"] is not None:
            by_topic.setdefault(rec["_topic"], []).append(rec)
    for topic, rs in by_topic.items():
        if len({str(x["_val"]) for x in rs}) < 2:
            continue
        rs = sorted(rs, key=lambda x: _ts(x["published_at"]))
        latest, older = rs[-1], [x for x in rs[:-1] if str(x["_val"]) != str(rs[-1]["_val"])]
        pub = _ts(latest["published_at"])
        if all(_ts(x["published_at"]) < pub and latest["_rank"] <= x["_rank"] for x in older):
            for x in older:
                x.update(verified=False, source_status="superseded",
                         reason_not_applied=f"superseded by a later report published {_iso(pub)}")
            latest["uncertainty"] = "; ".join(filter(None, [latest.get("uncertainty")] + [
                f"Earlier report ({_iso(_ts(x['published_at']))}): {x['_val']}" for x in older]))
        else:
            vals = sorted({str(x["_val"]) for x in rs})
            for x in rs:
                x["contradiction"] = True
                x["uncertainty"] = f"Sources disagree on {topic[1]}: " + " vs ".join(vals)
    out = []
    for rec in recs:
        for k in ("_topic", "_val", "_rank"):
            rec.pop(k)
        out.append(normalize_record(rec))
    return out


# --------------------------------------------------------------------------- #
# execution adapter: candidate row + run receipt
# --------------------------------------------------------------------------- #

#: (factor_id/stage, column, category, unit, public description)
_STAGE_COLUMNS = (
    ("realloc_volume", "realloc_mult", "availability_adjustment",
     "volume from a teammate ruled OUT (candidates.apply_reallocation)"),
    ("realloc_efficiency", "realloc_eff_mult", "availability_adjustment",
     "efficiency from a teammate ruled OUT (candidates.apply_reallocation)"),
    ("backup_qb", "backup_qb_adj", "qb_news",
     "backup-QB passing efficiency on receiving yards, receptions and passing yards "
     "(candidates.apply_backup_qb_adjustment)"),
    ("absence_qb", "absence_qb_mult", "availability_adjustment",
     "skill-leader absence on QB passing yards and pass attempts "
     "(candidates.apply_absence_qb_adjustment)"),
)


def records_from_forecast_row(row: Dict, receipt: Dict, as_of) -> List[Dict]:
    """Execution-derived records for one forecast row.

    ``receipt`` is the run's record of what executed::

        {"component": FORECAST_VERSION,
         "stages_executed": ["realloc_volume", "backup_qb", "absence_qb",
                             "dispersion", "game_script", ...],
         "primary_margin_source": "neutral",
         "ordering_component": "ml_ranker" | None,
         "ordering_features_populated": ["def_out_db", ...]}

    A stage missing from ``stages_executed`` is "not recorded", never zero.  A
    stage that ran but left no multiplier column on the row is a recorded
    neutral result (the pipeline stamps the column only where it changed the
    mean).  Nothing here reads the feature schema.
    """
    as_of_t = _as_of(as_of)
    ran = set(receipt.get("stages_executed") or ())
    comp = receipt.get("component")
    base = dict(entity_id=row.get("player_id"), entity_type="player", game_id=row.get("game_id"),
                team=row.get("team"), as_of=as_of_t, component=comp, model_version=comp,
                source_id="pipeline_execution", verified=True, measurement_kind="projected")
    out: List[Dict] = []
    for stage, col, cat, desc in _STAGE_COLUMNS:
        mult = _num(row.get(col))
        rec = dict(base, factor_id=stage, category=cat, feature_name=col,
                   rationale=f"Stage: {desc}.")
        if stage not in ran:
            rec.update(populated=False, consumed=False, measurement_kind="unavailable",
                       reason_not_applied="stage not recorded as executed in this run")
        elif mult is None or mult == 1.0:
            rec.update(consumed=True, evaluated_neutral=True, numerical_effect=1.0,
                       effect_unit="x projected mean", effect_method="executed_stage_multiplier",
                       observation="stage ran; no change")
        else:
            rec.update(consumed=True, numerical_effect=mult, effect_unit="x projected mean",
                       effect_method="executed_stage_multiplier", observation=f"x{mult:g}")
        out.append(normalize_record(rec))

    # game script: primary margin source decides whether the football margin is used
    if "game_script" in ran:
        src = row.get("margin_source") or receipt.get("primary_margin_source")
        prim = receipt.get("primary_margin_source")
        rec = dict(base, factor_id="game_script", category="game_script",
                   feature_name="forecast_margin", measurement_kind="projected",
                   observation=f"margin source: {src}")
        if src == "football" and prim == "football":
            rec.update(consumed=True, rationale="Team margin from prior game scores drives the "
                                                "pass/rush tilt.")
        elif src == "spread":
            rec.update(populated=False, verified=False, measurement_kind="unavailable",
                       reason_not_applied="sportsbook spread is not a permitted forecast input")
        else:
            rec.update(consumed=False, consumed_shadow=True,
                       rationale="The published projection uses a neutral game script; the "
                                 "score-based margin runs in shadow.")
        out.append(normalize_record(rec))

    if "dispersion" in ran:
        role = row.get("dispersion_role")
        rec = dict(base, factor_id="dispersion", category="dispersion",
                   feature_name="sd_conditional", observation=f"mean-conditional SD: {role or 'none'}")
        if role == "primary":
            rec.update(consumed=True, rationale="SD scales with the projected mean for this market.")
        elif role == "shadow":
            rec.update(consumed_shadow=True, rationale="Mean-conditional SD is shadow for this market; "
                                                       "the pooled SD is published.")
        else:
            rec.update(consumed=True, evaluated_neutral=True, rationale="No fitted parameters; "
                                                                        "pooled SD kept.")
        out.append(normalize_record(rec))

    ordering = receipt.get("ordering_component")
    for feat in (receipt.get("ordering_features_populated") or ()) if ordering else ():
        if feat not in row:
            continue
        out.append(normalize_record(dict(
            base, factor_id=f"ordering:{feat}", category="ordering", feature_name=feat,
            component=ordering, value=row.get(feat), observation=f"{feat} = {row.get(feat)}",
            ordering_consumed=True, measurement_kind="observed",
            rationale="Read by the ordering model, which ranks rows; it does not set mean or SD.")))
    return out


# --------------------------------------------------------------------------- #
# descriptive history and schedule
# --------------------------------------------------------------------------- #

def split_context_record(*, entity_id: str, entity_type: str, game_id: str, as_of, split_kind: str,
                         stat: str, split_mean: Optional[float], split_n: int,
                         baseline_mean: Optional[float], baseline_n: int, cutoff: str,
                         games: Sequence[Dict], source_id: str, team: Optional[str] = None,
                         source_url: Optional[str] = None) -> Dict:
    """A descriptive split (venue, opponent, home/away, primetime, season, recent).

    Always ``context_only``: a split mean is history, not a model input.  The
    denominator and cutoff are part of the observation; ``games`` must all be on
    or before ``cutoff`` and before ``as_of``.
    """
    as_of_t = _as_of(as_of)
    cut = dt.date.fromisoformat(cutoff)
    if cut > as_of_t.date():
        raise FactorRecordError("split cutoff is after as_of")
    for g in games:
        if dt.date.fromisoformat(str(g["gameday"])[:10]) > cut:
            raise FactorRecordError(f"game {g.get('game_id')} is after the cutoff {cutoff}")
    if split_n != len(games) and games:
        raise FactorRecordError("split_n must equal the number of listed games")
    base = (f"baseline {baseline_mean:.1f}, n={baseline_n}" if baseline_mean is not None
            else f"baseline n={baseline_n}")
    if split_n < MIN_SPLIT_GAMES or split_mean is None:
        obs = f"{split_kind}: Insufficient evidence (n={split_n}) for {stat} ({base}; through {cutoff})"
    else:
        obs = f"{split_kind}: {split_mean:.1f} {stat}, n={split_n} ({base}; through {cutoff})"
    return normalize_record(dict(
        factor_id=f"split:{split_kind}:{entity_id}", category=split_kind if split_kind in CATEGORY_LABELS
        else "season_form", entity_id=entity_id, entity_type=entity_type, team=team, game_id=game_id,
        as_of=as_of_t, observation=obs, value=split_mean, unit=stat, source_id=source_id,
        source_url=source_url, verified=True, measurement_kind="observed",
        support_games=split_n, support_scope="historical", evidence_games=[g["game_id"] for g in games],
        rationale="Descriptive history; not used by the projection.",
        uncertainty=None if split_n >= MIN_SPLIT_GAMES else "Too few games to separate from noise."))


def _local_hour(kickoff: str) -> int:
    t = _ts(kickoff)
    if t is None:
        raise FactorRecordError(f"kickoff {kickoff!r} needs a venue-local UTC offset")
    return t.hour


def schedule_context_record(team: str, games: Sequence[Dict], as_of,
                            sources: Sequence[Dict]) -> Dict:
    """Upcoming primetime games for ``team``, verified only by a team/league source.

    ``games``: [{game_id, week, kickoff (ISO with the VENUE-local offset), network}].
    ``sources``: [{source_url, source_tier, fetched_at, agrees}].  A schedule
    fact is context: it supports no performance multiplier.
    """
    as_of_t = _as_of(as_of)
    prime = [g for g in games if _local_hour(g["kickoff"]) >= PRIMETIME_LOCAL_HOUR]
    official = [s for s in sources if s.get("source_tier") in PRIMARY_TIERS and s.get("agrees")
                and (_ts(s.get("fetched_at")) or as_of_t) <= as_of_t]
    disagree = [s for s in sources if s.get("agrees") is False]
    src = (official or list(sources) or [{}])[0]
    weeks = ", ".join(f"Wk{g['week']} {g.get('network') or '?'}" for g in prime)
    rec = normalize_record(dict(
        factor_id=f"schedule:{team}:primetime_next{len(games)}", category="primetime",
        entity_id=team, entity_type="team", team=team, game_id=games[0]["game_id"] if games else None,
        as_of=as_of_t, observation=f"{len(prime)} of next {len(games)} games in primetime ({weeks})",
        value=len(prime), unit="games", source_url=src.get("source_url"),
        fetched_at=src.get("fetched_at"), source_id=src.get("source_id"),
        verified=bool(official) and not disagree, measurement_kind="observed",
        reliability=("Confirmed by team source" if official and official[0]["source_tier"] == "team_official"
                     else "Confirmed by league source" if official else "Not confirmed by team or league"),
        rationale="Schedule fact; no performance adjustment is supported by the schedule alone.",
        uncertainty=("Sources disagree on the schedule" if disagree
                     else "Kickoff times can change after publication."),
        contradiction=bool(disagree), expires_at=None))
    rec["evidence_games"] = [g["game_id"] for g in prime]
    return rec


# --------------------------------------------------------------------------- #
# public labels, panel, HTML
# --------------------------------------------------------------------------- #

def _fmt_clock(r: Dict) -> str:
    parts = []
    for k, word in (("published_at", "published"), ("observed_at", "observed"), ("fetched_at", "fetched")):
        if r.get(k):
            parts.append(f"{word} {r[k][:16].replace('T', ' ')} UTC")
    return "; ".join(parts) or "no source clock"


def public_label(r: Dict) -> Dict:
    """Record -> short bettor-facing strings.  Values are interpolated, never computed."""
    st = r["status"]
    if st == "numeric_applied":
        e = r.get("numerical_effect")
        if e is None:
            status_label = NOT_ISOLATED
        elif (r.get("effect_unit") or "").startswith("x"):
            status_label = (f"Used in projection: x{e:g} on the projected mean"
                            + (" (as executed)" if r.get("effect_method") == "executed_stage_multiplier" else ""))
        else:
            status_label = f"Used in projection: {e:g} {r.get('effect_unit') or ''}".rstrip()
    elif st == "context_only" and r.get("ordering_consumed"):
        status_label = ORDERING_ONLY_LABEL
    else:
        status_label = STATUS_LABELS[st]
    mk = r.get("measurement_kind")
    measurement = {"observed": "Observed", "projected": "Model output", "unavailable": "No data",
                   "proxy": f"Proxy ({r.get('proxy_name')})"}.get(mk, "Unknown")
    sg = r.get("support_games")
    if sg is None:
        support = None
    elif r.get("support_scope") == "current_season":
        support = f"{sg:g} games this season"
    elif r.get("support_scope") == "historical":
        support = f"{sg:g} historical games; not current-role evidence"
    else:
        support = f"{sg:g} games"
    caution = []
    if r.get("contradiction") and r.get("uncertainty"):
        caution.append(r["uncertainty"])
    elif r.get("uncertainty"):
        caution.append(r["uncertainty"])
    if st == "numeric_applied" and not r.get("verified"):
        caution.append("input source not verified")
    detail = r.get("rationale") or ""
    if st not in ("numeric_applied", "considered_no_change") and r.get("reason_not_applied"):
        detail = (detail + " " if detail else "") + f"Why not used: {r['reason_not_applied']}."
    obs = r.get("observation")
    if r.get("text_withheld"):
        obs = "[claim text withheld]"
    return {
        "factor_id": r["factor_id"], "category": r["category"],
        "category_label": CATEGORY_LABELS.get(r["category"], r["category"]),
        "entity": r.get("entity_id"), "status": st, "status_label": status_label,
        "observation": obs if obs is not None else "No value recorded",
        "measurement": measurement, "support": support,
        "reliability": r.get("reliability"),
        "source": ({"url": r.get("source_url"), "title": r.get("source_title") or r.get("source_id")}
                   if r.get("source_url") or r.get("source_id") else None),
        "clock": _fmt_clock(r), "detail": detail.strip(),
        "caution": "; ".join(caution) or None,
    }


def check_copy(text: str) -> None:
    low = text.lower()
    hits = [t for t in BANNED_TERMS + _CERTAINTY if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", low)]
    if hits:
        raise UnsafeCopy(f"public copy contains unsupported certainty/imperative terms: {hits}")


def select_for_card(records: Iterable[Dict], card: Dict) -> List[Dict]:
    """Records for one card: the player, their team, and game-level records."""
    pid, gid, team = card.get("player_id"), card.get("game_id"), card.get("team")
    out = []
    for r in records:
        if r.get("game_id") not in (None, gid):
            continue
        et = r.get("entity_type")
        if (et == "player" and r.get("entity_id") == pid) or \
           (et == "team" and team and r.get("entity_id") == team) or et in ("game", None):
            out.append(r)
    return out


def build_panel(records: Iterable[Dict], as_of) -> Dict:
    """Structured evidence panel grouped by category.  Raises UnsafeCopy on banned copy."""
    as_of_t = _as_of(as_of)
    labels = [public_label(r) for r in records]
    for lab in labels:
        check_copy(" ".join(str(v) for k, v in lab.items() if isinstance(v, str)))
    groups = []
    for cat in CATEGORY_ORDER + tuple(sorted({l["category"] for l in labels} - set(CATEGORY_ORDER))):
        items = [l for l in labels if l["category"] == cat]
        if items:
            groups.append({"category": cat, "label": CATEGORY_LABELS.get(cat, cat),
                           "items": sorted(items, key=lambda l: (STATUSES.index(l["status"]),
                                                                 l["factor_id"]))})
    return {"schema_version": SCHEMA_VERSION, "as_of": _iso(as_of_t), "groups": groups,
            "counts": {s: sum(l["status"] == s for l in labels) for s in STATUSES},
            "legend": dict(STATUS_LABELS)}


def render_panel_html(panel: Dict) -> str:
    """Minimal escaped HTML fragment for the card surface (no scripts, no styles)."""
    e = lambda x: html.escape("" if x is None else str(x), quote=True)
    parts = [f'<div class="fe-panel" data-schema="{e(panel.get("schema_version"))}">',
             f'<div class="fe-head">What the model knew and used (as of {e(panel.get("as_of"))})</div>']
    if not panel.get("groups"):
        parts.append('<div class="fe-empty">No factor records for this card.</div></div>')
        return "".join(parts)
    for g in panel["groups"]:
        parts.append(f'<div class="fe-group"><div class="fe-cat">{e(g["label"])}</div><ul>')
        for it in g["items"]:
            src = it.get("source") or {}
            url = src.get("url") or ""
            link = (f' <a href="{e(url)}" rel="nofollow noopener">{e(src.get("title") or url)}</a>'
                    if url.startswith("https://") else (f" {e(src.get('title'))}" if src.get("title") else ""))
            parts.append(
                f'<li data-status="{e(it["status"])}"><b>{e(it["status_label"])}</b> &middot; '
                f'{e(it["observation"])} <span class="fe-m">[{e(it["measurement"])}'
                + (f"; {e(it['support'])}" if it.get("support") else "") + "]</span>"
                + (f' <span class="fe-r">{e(it["reliability"])}</span>' if it.get("reliability") else "")
                + f'<div class="fe-d">{e(it["detail"])}</div>'
                + (f'<div class="fe-c">Caution: {e(it["caution"])}</div>' if it.get("caution") else "")
                + f'<div class="fe-s">Source:{link or " none"}; {e(it["clock"])}</div></li>')
        parts.append("</ul></div>")
    parts.append("</div>")
    return "".join(parts)
