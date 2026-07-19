"""Phase 8 acceptance tests -- the explainability layer's honesty contract.

These are the tests named in the phase brief, plus the ones that make its
invariants enforceable rather than aspirational. The governing idea is
CALIBRATION, NOT PERSUASION: this tool has not proven edge, so a card that
makes a thin case feel overwhelming is the most expensive bug it could ship.

Offline and deterministic.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import evidence as ev            # noqa: E402
from nflvalue import explain                   # noqa: E402
from nflvalue import explain_render as rn      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEEKLY = os.path.join(ROOT, "data", "weekly_props.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(WEEKLY),
    reason="data/weekly_props.json is a pipeline artifact; run pipeline_weekly first")


def _leans():
    with open(WEEKLY, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [(g, l) for g in payload["games"] for l in g["leans"]]


def _refs():
    return ev.load_reference_levels()


# --------------------------------------------------------------------------- #
# 1. Ledger contributions reconcile to the final projection
# --------------------------------------------------------------------------- #
def test_every_ledger_reconciles_to_the_shipped_projection():
    """The load-bearing test. If a multiplier is applied to the number but not
    registered in ADJUSTMENT_COLUMNS, the replay lands elsewhere and this
    fails -- which is what makes "a silent adjustment is a bug" mechanical."""
    failures = []
    for _game, lean in _leans():
        try:
            explain.build_ledger(lean, refs=_refs())
        except explain.LedgerReconciliationError as exc:
            failures.append(str(exc))
    assert not failures, f"{len(failures)} ledger(s) failed to reconcile:\n" + \
                         "\n".join(failures[:5])


def test_reconciliation_drift_stays_inside_the_rounding_bound():
    for _game, lean in _leans():
        led = explain.build_ledger(lean, refs=_refs())
        assert led.reconciliation_drift <= led.reconciliation_bound, (
            f"{led.name} {led.market}: drift {led.reconciliation_drift} "
            f"exceeded rounding bound {led.reconciliation_bound}")


def test_an_unregistered_adjustment_is_caught(monkeypatch):
    """MUTATION: apply a multiplier to the shipped mean that the ledger knows
    nothing about. This is the exact shape of the bug the reconciliation check
    exists to catch, so it must fail loudly."""
    _game, lean = _leans()[0]
    poisoned = dict(lean)
    poisoned["mean"] = round(float(lean["mean"]) * 1.15, 3)   # silent +15%
    with pytest.raises(explain.LedgerReconciliationError):
        explain.build_ledger(poisoned, refs=_refs())


def test_log_shares_are_an_exact_decomposition():
    """Shares sum to 1 over the non-baseline drivers -- the order-independent
    view of the same decomposition."""
    for _game, lean in _leans():
        led = explain.build_ledger(lean, refs=_refs())
        shares = led.log_shares()
        if shares:
            assert math.isclose(sum(shares.values()), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# 2. A synthetic-line pick can never display an edge
# --------------------------------------------------------------------------- #
def test_synthetic_line_pick_never_exposes_an_edge():
    """The accuracy protocol forbids deriving any edge/ROI/CLV claim from a
    synthetic reference line. `Ledger.edge` is the enforcement point."""
    seen_synthetic = False
    for _game, lean in _leans():
        led = explain.build_ledger(lean, refs=_refs())
        if led.is_synthetic_line:
            seen_synthetic = True
            assert led.edge(lean) is None, (
                f"{led.name} {led.market} exposed an edge against a synthetic line")
    assert seen_synthetic, "fixture contained no synthetic-line picks to test"


def test_synthetic_line_is_disclosed_in_the_prose():
    for game, lean in _leans():
        led = explain.build_ledger(lean, refs=_refs())
        if not led.is_synthetic_line:
            continue
        case = rn.render_case(ev.attach_evidence(led.to_dict(), _refs()),
                              screened=len(game["leans"]),
                              screened_n=game.get("screened_n"))
        assert "SYNTHETIC" in case["text"], "synthetic line not disclosed in copy"
        return


# --------------------------------------------------------------------------- #
# 3. Show the denominator
# --------------------------------------------------------------------------- #
def test_screen_count_is_rendered_with_its_denominator():
    game, lean = _leans()[0]
    led = ev.attach_evidence(explain.build_ledger(lean, refs=_refs()).to_dict(), _refs())
    case = rn.render_case(led, screened=len(game["leans"]),
                          screened_n=game.get("screened_n"))
    assert f"of {game['screened_n']} candidates screened" in case["text"]


# --------------------------------------------------------------------------- #
# 4. Never render a one-sided case
# --------------------------------------------------------------------------- #
def test_counter_case_block_is_always_present():
    """Even when nothing opposes, the block renders and says so -- an empty
    counter-case must read as 'few factors were measured', never as silence."""
    for game, lean in _leans():
        led = ev.attach_evidence(explain.build_ledger(lean, refs=_refs()).to_dict(), _refs())
        case = rn.render_case(led, screened=len(game["leans"]),
                              screened_n=game.get("screened_n"))
        assert case["blocks"]["counter_case"], f"{lean['name']}: no counter-case block"


def test_levels_without_a_reference_are_not_treated_as_directional():
    """A catch rate of 0.568 is a unit conversion, not an argument for the
    under. Manufacturing a counter-case out of arithmetic would be false
    balance -- the opposite of the calibration goal."""
    c = explain.Contribution(
        key="efficiency", label="Efficiency", stage="efficiency",
        multiplier=0.568, value_before=10.0, value_after=5.68,
        unit="receptions", kind="level", reference=None,
        provenance=explain.Provenance("test"))
    assert c.direction == "level"

    c.reference = 0.626
    assert c.direction == "down", "a measured reference should make it directional"


# --------------------------------------------------------------------------- #
# 5. Evidence: no strength word without n
# --------------------------------------------------------------------------- #
def test_no_strength_claim_without_a_sample_size_or_explicit_disclosure():
    for game, lean in _leans():
        led = ev.attach_evidence(explain.build_ledger(lean, refs=_refs()).to_dict(), _refs())
        for sentence in rn.render_evidence(led):
            text = sentence["text"]
            if any(w in text for w in ("evidence is strong", "evidence is moderate",
                                       "evidence is thin", "evidence is unproven")):
                assert ("n=" in text or "n not published" in text), (
                    f"strength claim with no sample size: {text}")


def test_every_evidence_entry_states_its_interval_status():
    for _game, lean in _leans():
        led = ev.attach_evidence(explain.build_ledger(lean, refs=_refs()).to_dict(), _refs())
        for c in led["contributions"]:
            e = c["evidence"]
            assert e["interval_status"] in ("computed", "not_recomputed",
                                            "not_applicable")
            if e["interval_status"] == "computed":
                assert e["interval"] and len(e["interval"]) == 2


def test_thin_evidence_is_distinguishable_from_strong():
    """Screenshot-diff proxy: the grade string differs, so any renderer keyed
    on it produces different output."""
    thin = ev.grade_from(n=20, interval=[0.48, 0.85], effect_unit="proportion")
    strong = ev.grade_from(n=800, interval=[0.55, 0.61],
                           effect_unit="proportion", replicated=True)
    assert thin == "thin" and strong == "strong"
    assert thin != strong


def test_a_lucky_small_sample_grades_worse_than_a_boring_large_one():
    """70% on n=20 must look WEAKER than 62% on n=800 -- the stated goal."""
    lucky = ev.wilson_interval(14, 20)          # 70%
    boring = ev.wilson_interval(496, 800)       # 62%
    assert lucky[0] < boring[0], (
        f"small-sample lower bound {lucky[0]} should sit below {boring[0]}")
    assert ev.grade_from(20, lucky, "proportion") == "thin"


def test_narrative_factors_render_as_measured_zero():
    """Birthdays and revenge were tested and came back at ~zero. They must
    appear as measured, with n and a CI -- not as colour, and not omitted."""
    for key in ("birthday", "revenge"):
        e = ev.MEASURED_ZERO[key]
        assert e.grade == "unproven"
        assert e.n and e.n > 1000
        assert e.interval and len(e.interval) == 2
        assert e.interval_status == "computed"


# --------------------------------------------------------------------------- #
# 6. Vocabulary
# --------------------------------------------------------------------------- #
def test_generated_copy_never_contains_imperative_betting_language():
    for game, lean in _leans():
        led = ev.attach_evidence(explain.build_ledger(lean, refs=_refs()).to_dict(), _refs())
        rn.render_case(led, screened=len(game["leans"]),
                       screened_n=game.get("screened_n"))   # raises on violation


@pytest.mark.parametrize("bad", [
    "This is a lock at this number.",
    "Hammer the over here.",
    "Max bet material.",
    "The evidence is overwhelming.",
    "A no-brainer play.",
])
def test_vocabulary_check_actually_catches_banned_language(bad):
    """The banned-word test must have teeth -- verified by feeding it copy that
    should fail."""
    with pytest.raises(rn.VocabularyViolation):
        rn.check_vocabulary(bad)


def test_vocabulary_check_permits_the_sanctioned_register():
    rn.check_vocabulary(
        "The lean is over. The case is that his usage share rose. "
        "Evidence is thin at n=20, 95% CI 0.481-0.855.")


# --------------------------------------------------------------------------- #
# 7. Prose invents no numbers
# --------------------------------------------------------------------------- #
def test_evidence_notes_are_frozen_registry_text_not_generated():
    """Evidence notes are hand-written provenance ("~31% efficiency loss",
    "capped x1.35"). They legitimately contain constants that are NOT ledger
    values, so they are excluded from the generated-number check above -- which
    means they need their own guard: they must appear VERBATIM in the registry,
    so no caller can inject or reword a factual claim at render time."""
    registry = set()
    for key in ("team_volume", "usage_share", "efficiency", "game_script",
                "opp_factor", "realloc_mult", "realloc_eff_mult",
                "backup_qb_adj", "absence_qb_mult"):
        note = ev.evidence_for(key).note
        if note:
            registry.add(note)
    for e in ev.MEASURED_ZERO.values():
        if e.note:
            registry.add(e.note)

    for _game, lean in _leans()[:10]:
        led = ev.attach_evidence(explain.build_ledger(lean, refs=_refs()).to_dict(), _refs())
        for c in led["contributions"]:
            note = (c.get("evidence") or {}).get("note")
            if note:
                assert note in registry, f"note not from the frozen registry: {note!r}"


def _strip_notes(text: str) -> str:
    """Remove frozen registry notes so the number check sees only COMPUTED
    prose. The notes are guarded separately by the test above."""
    for key in ("team_volume", "usage_share", "efficiency", "game_script",
                "opp_factor", "realloc_mult", "realloc_eff_mult",
                "backup_qb_adj", "absence_qb_mult"):
        note = ev.evidence_for(key).note
        if note:
            text = text.replace(note, "")
    return text


def test_every_number_in_the_prose_comes_from_the_ledger():
    """The renderer interpolates pre-formatted strings and never computes, so
    every numeric token in the COMPUTED copy must be reproducible from a ledger
    value through `fmt`. This is the `synthesis.py` contract applied to prose.

    Static registry notes are excluded and covered by
    ``test_evidence_notes_are_frozen_registry_text_not_generated``.
    """
    for game, lean in _leans()[:25]:
        led = ev.attach_evidence(explain.build_ledger(lean, refs=_refs()).to_dict(), _refs())
        case = rn.render_case(led, screened=len(game["leans"]),
                              screened_n=game.get("screened_n"))

        allowed = {str(len(game["leans"])), str(game.get("screened_n")), "95"}
        for c in led["contributions"]:
            for v in (c.get("multiplier"), c.get("value_after"),
                      c.get("value_before"), c.get("delta"), c.get("reference")):
                if v is None:
                    continue
                for dp in (0, 1, 3):
                    allowed.add(rn.fmt(v, dp))
                    allowed.add(rn.fmt(abs(v), dp))
            for extra in ("reference_n", "reference_ci"):
                val = (c.get("inputs") or {}).get(extra)
                if isinstance(val, (int, float)):
                    allowed.add(str(val))
                    allowed.add(f"{val:,}")
                elif isinstance(val, list):
                    for x in val:
                        allowed.update({rn.fmt(x, 3), str(x)})
            e = c.get("evidence") or {}
            if e.get("n"):
                allowed.update({str(e["n"]), f"{e['n']:,}"})
            for x in (e.get("interval") or []):
                allowed.update({rn.fmt(x, 3), str(x)})
        if led.get("line") is not None:
            allowed.add(rn.fmt(led["line"], 1))

        allowed_floats = set()
        for a in allowed:
            try:
                allowed_floats.add(round(float(str(a).replace(",", "")), 6))
            except ValueError:
                pass

        for tok in rn.numbers_in(_strip_notes(case["text"])):
            val = round(float(tok.replace(",", "")), 6)
            assert tok in allowed or val in allowed_floats, (
                f"prose contains {tok!r}, which is not a ledger value "
                f"({lean['name']} {lean['market']})")


def test_sentences_are_all_traced_to_a_ledger_entry():
    for game, lean in _leans()[:25]:
        led = ev.attach_evidence(explain.build_ledger(lean, refs=_refs()).to_dict(), _refs())
        case = rn.render_case(led, screened=len(game["leans"]),
                              screened_n=game.get("screened_n"))
        for block in case["blocks"].values():
            for sentence in block:
                assert sentence.get("entry"), f"untraced sentence: {sentence}"


# --------------------------------------------------------------------------- #
# 8. The ranking ledger is kept separate from the number
# --------------------------------------------------------------------------- #
def test_ranking_ledger_declares_it_does_not_move_the_projection():
    _game, lean = _leans()[0]
    ranking = explain.build_ranking_ledger(lean)
    assert ranking["affects_projection"] is False
    assert "did not change the projected number" in ranking["disclosure"]


def test_ml_importances_are_labelled_global_not_per_pick():
    """There is no per-instance attribution available without a new
    dependency, and inventing one would be a fabrication."""
    _game, lean = _leans()[0]
    ranking = explain.build_ranking_ledger(
        lean, ml_importances=[{"feature": "red_zone_share", "importance": 0.08}])
    block = ranking["ml_feature_importance"]
    assert block["scope"] == "global"
    assert "NOT an attribution for this pick" in block["caveat"]


# --------------------------------------------------------------------------- #
# 9. Determinism
# --------------------------------------------------------------------------- #
def test_bootstrap_interval_is_deterministic():
    vals = [1.0, 1.2, 0.8, 1.1, 0.95, 1.3, 0.7, 1.05] * 20
    blocks = [f"KC_{i % 8}" for i in range(len(vals))]
    first = ev.block_bootstrap_mean(vals, blocks)
    for _ in range(3):
        assert ev.block_bootstrap_mean(vals, blocks) == first


def test_rendering_is_deterministic():
    game, lean = _leans()[0]
    led = ev.attach_evidence(explain.build_ledger(lean, refs=_refs()).to_dict(), _refs())
    first = rn.render_case(led, screened=len(game["leans"]),
                           screened_n=game.get("screened_n"))["text"]
    for _ in range(3):
        again = rn.render_case(led, screened=len(game["leans"]),
                               screened_n=game.get("screened_n"))["text"]
        assert again == first
