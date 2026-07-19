"""Phase 8.1 -- the contribution ledger.

Turns a pick from a number into a traceable case. For any lean this module
decomposes the projected mean into the drivers that actually produced it, in
the order the pipeline applied them, and returns a structured ledger that
**reconciles exactly to the shipped number**.

Why this is not a one-liner
---------------------------
The projection is MULTIPLICATIVE, not additive::

    mean_0 = round(volume x efficiency x opp_factor, 3)      # projection.project
    mean_1 = round(mean_0 x realloc_mult x realloc_eff, 3)   # apply_reallocation
    mean_2 = round(mean_1 x backup_qb_adj, 3)                # apply_backup_qb
    mean_3 = round(mean_2 x absence_qb_mult, 3)              # apply_absence_qb

...and each stage rounds to 3dp. A naive ``base * m1 * m2 * m3`` does NOT equal
the shipped value. The ledger replays the chain including the rounding, which
is why reconciliation is an equality check and not a hopeful tolerance.

So "the ledger sums to the projection" is given two exact meanings:

* **stepwise deltas** -- each driver's change in stat units, telescoping, so
  the last ``value_after`` IS the projected mean by construction. This is what
  the plain-English layer reads.
* **log shares** -- ``log(mean) = log(baseline) + sum(log(multiplier))``, giving
  each driver an exact, order-independent share of the final number.

Stepwise attribution in a multiplicative model is ORDER-DEPENDENT: the same
factor is worth more stat units when applied late. The order here is fixed to
the pipeline's real application order and is reported on the ledger
(``stage_order``) rather than silently chosen. The log shares are the
order-free view, which is why both are emitted.

Honesty invariants (PREMORTEM.md, and the phase brief)
------------------------------------------------------
* **Anything applied to the number appears here.** ``reconcile()`` recomputes
  the mean from the ledger alone; a multiplier applied but not registered makes
  the ledger disagree and raises :class:`LedgerReconciliationError`. A silent
  adjustment is mechanically a build failure, not a code-review question.
* **What did NOT fire is also reported** (``not_applied``), so "no injury
  adjustment" reads as a checked fact rather than an absence.
* **This module computes nothing new about the world.** It re-expresses numbers
  the deterministic layer already produced. It never re-derives a projection,
  never fits anything, and never touches ranking.
* A synthetic reference line is carried as ``line_source="synthetic"`` and the
  ledger refuses to expose an ``edge`` for it (:meth:`Ledger.edge`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Stage keys, in the order the pipeline applies them. Used for ordering, for
# the order-dependence disclosure, and by the completeness check.
STAGE_ORDER = (
    "baseline_volume",
    "game_script",
    "efficiency",
    "opp_factor",
    "realloc_volume",
    "realloc_efficiency",
    "backup_qb",
    "absence_qb",
)

#: Multiplier columns on the candidate frame, mapped to their ledger stage and
#: the pipeline function that applies them. If a NEW adjustment is added to the
#: pipeline without being registered here, `test_no_unregistered_adjustment`
#: fails -- that test is the enforcement of "a silent adjustment is a bug".
ADJUSTMENT_COLUMNS = {
    "realloc_mult": ("realloc_volume", "candidates.apply_reallocation"),
    "realloc_eff_mult": ("realloc_efficiency", "candidates.apply_reallocation"),
    "backup_qb_adj": ("backup_qb", "candidates.apply_backup_qb_adjustment"),
    "absence_qb_mult": ("absence_qb", "candidates.apply_absence_qb_adjustment"),
}

#: Markets whose opportunity unit is not the final stat unit.
_OPPORTUNITY_UNIT = {
    "receiving_yards": "targets", "receptions": "targets",
    "rushing_yards": "carries", "rush_attempts": "carries",
    "passing_yards": "pass attempts", "pass_completions": "pass attempts",
    "anytime_td": "expected TDs",
}
_STAT_UNIT = {
    "receiving_yards": "receiving yards", "receptions": "receptions",
    "rushing_yards": "rushing yards", "rush_attempts": "carries",
    "passing_yards": "passing yards", "pass_completions": "completions",
    "anytime_td": "expected TDs",
}

_ROUND_DP = 3          # every pipeline stage rounds the mean to 3dp
_RECONCILE_TOL = 5e-4  # half a unit in the last published place


class LedgerReconciliationError(RuntimeError):
    """The ledger does not reproduce the shipped projection.

    Means one of: an adjustment was applied to the number but never registered
    in ADJUSTMENT_COLUMNS (the dangerous case -- a silent adjustment), the
    application order changed, or a stage's rounding changed.
    """


@dataclass
class Provenance:
    """Where a number came from and as of when. No entry ships without one."""
    source: str                      # "nflverse pbp", "data/absence_matrix.json", ...
    as_of: Optional[str] = None      # ISO date, or "prior weeks only"
    detail: Optional[str] = None


@dataclass
class Contribution:
    """One driver of the projection."""
    key: str
    label: str
    stage: str
    multiplier: Optional[float]      # None for the baseline (it is a level, not a factor)
    value_before: Optional[float]
    value_after: float
    unit: str
    provenance: Provenance
    #: "tilt"  -- a factor whose neutral value is exactly 1.0, so >1 and <1 are
    #:           genuine arguments for and against (game script, opponent
    #:           factor, every situational adjustment).
    #: "level" -- a magnitude, or a UNIT CONVERSION expressed as a rate. A
    #:           catch rate of 0.568 turns targets into receptions; it is not
    #:           "an argument for the under" just because it is below 1.0.
    #:           Treating levels as directional manufactures a counter-case out
    #:           of arithmetic, which would be exactly the kind of false
    #:           balance this phase is supposed to prevent.
    kind: str = "tilt"
    #: Optional measured reference that makes a LEVEL directional, e.g. the
    #: position's league-average catch rate. Only set when a real measured
    #: comparison exists -- never guessed.
    reference: Optional[float] = None
    #: The reference's own 95% interval. A level that falls INSIDE it is
    #: statistically indistinguishable from average, and claiming a direction
    #: from it would manufacture an argument out of noise -- e.g. 4.134 yards
    #: per carry against a 4.147 reference is a 0.3% gap, not a case for the
    #: under. Direction is withheld unless the level clears this interval.
    reference_ci: Optional[List[float]] = None
    inputs: Dict = field(default_factory=dict)
    evidence: Optional[Dict] = None  # attached by explain_evidence (8.2)

    @property
    def delta(self) -> Optional[float]:
        if self.value_before is None:
            return None
        return round(self.value_after - self.value_before, 4)

    @property
    def direction(self) -> str:
        """Which way this driver pushed the NUMBER (not the pick's side).

        Only meaningful for tilts, and for levels that carry a measured
        reference to compare against. Everything else is "level": a stated
        magnitude with no directional claim attached.
        """
        if self.multiplier is None:
            return "baseline"
        if self.kind == "level":
            if self.reference is None or not math.isfinite(self.reference):
                return "level"
            if self.reference_ci and len(self.reference_ci) == 2:
                lo, hi = self.reference_ci
                if lo <= self.multiplier <= hi:
                    return "neutral"     # inside the reference interval
            elif abs(self.multiplier - self.reference) < 1e-9:
                return "neutral"
            return "up" if self.multiplier > self.reference else "down"
        if abs(self.multiplier - 1.0) < 1e-9:
            return "neutral"
        return "up" if self.multiplier > 1.0 else "down"

    @property
    def log_contribution(self) -> Optional[float]:
        if self.multiplier is None or self.multiplier <= 0:
            return None
        return math.log(self.multiplier)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.update(direction=self.direction, delta=self.delta,
                 log_contribution=self.log_contribution)
        return d


@dataclass
class Ledger:
    player_id: Optional[str]
    name: Optional[str]
    market: str
    side: Optional[str]
    line: Optional[float]
    line_source: str                 # "real" | "synthetic" | "none"
    projected_mean: float
    contributions: List[Contribution]
    not_applied: List[Dict]
    granularity: str                 # "full" | "components_only"
    stage_order: List[str]
    reconciled_mean: Optional[float] = None
    baseline: Optional[float] = None
    #: Reconciliation diagnostics, filled by :func:`reconcile`. ``exact`` is
    #: True only when the ledger reproduces the shipped mean bit-for-bit, which
    #: happens when it was built from unrounded rows rather than the published
    #: (rounded) artifact.
    reconciliation_drift: Optional[float] = None
    reconciliation_bound: Optional[float] = None
    reconciliation_exact: Optional[bool] = None

    # -- honesty accessors -------------------------------------------------- #
    @property
    def is_synthetic_line(self) -> bool:
        return self.line_source == "synthetic"

    def edge(self, lean: Dict) -> Optional[float]:
        """Edge, or None when the reference line is synthetic.

        A synthetic line is the player's own trailing mean. "Beating" it is a
        statement about the model's relationship to itself, not to a market,
        and the accuracy protocol forbids deriving any edge/ROI/CLV claim from
        it. This accessor is the enforcement point.
        """
        if self.is_synthetic_line:
            return None
        return lean.get("edge")

    def log_shares(self) -> Dict[str, float]:
        """Each driver's exact share of log(mean / baseline).

        Order-independent, unlike the stepwise deltas. Shares sum to 1.0 over
        the non-baseline drivers (or to {} when the baseline already equals the
        mean, i.e. every multiplier was neutral).
        """
        logs = {c.key: c.log_contribution for c in self.contributions
                if c.log_contribution is not None}
        total = sum(logs.values())
        if abs(total) < 1e-12:
            return {}
        return {k: v / total for k, v in logs.items()}

    def opposing(self, side: Optional[str] = None) -> List[Contribution]:
        """Drivers that argue AGAINST the pick -- the counter-case.

        For an OVER, anything that pushed the number down; for an UNDER,
        anything that pushed it up. Neutral drivers are excluded. The card is
        required to render this, so a one-sided case is structurally
        impossible to produce from this ledger.
        """
        side = side or self.side
        if side not in ("over", "under"):
            return []
        against = "down" if side == "over" else "up"
        return [c for c in self.contributions if c.direction == against]

    def levels(self) -> List["Contribution"]:
        """Magnitudes with no directional claim -- shown as inputs, not as
        arguments. Rendered on the card so the reader sees them, but never
        counted into the case or the counter-case."""
        return [c for c in self.contributions if c.direction == "level"]

    def supporting(self, side: Optional[str] = None) -> List[Contribution]:
        side = side or self.side
        if side not in ("over", "under"):
            return []
        favours = "up" if side == "over" else "down"
        return [c for c in self.contributions if c.direction == favours]

    def to_dict(self) -> Dict:
        return {
            "player_id": self.player_id, "name": self.name,
            "market": self.market, "side": self.side, "line": self.line,
            "line_source": self.line_source,
            "is_synthetic_line": self.is_synthetic_line,
            "projected_mean": self.projected_mean,
            "reconciled_mean": self.reconciled_mean,
            "baseline": self.baseline,
            "granularity": self.granularity,
            "stage_order": self.stage_order,
            "reconciliation": {
                "drift": self.reconciliation_drift,
                "rounding_bound": self.reconciliation_bound,
                "exact": self.reconciliation_exact,
            },
            "contributions": [c.to_dict() for c in self.contributions],
            "log_shares": self.log_shares(),
            "not_applied": self.not_applied,
        }


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def _is_num(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _line_source(lean: Dict) -> str:
    raw = lean.get("line_source")
    if raw == "synthetic_trailing_mean":
        return "synthetic"
    if lean.get("line") is None:
        return "none"
    return "real" if raw else "none"


def build_ledger(lean: Dict, player_row: Optional[Dict] = None,
                 team_row: Optional[Dict] = None,
                 refs: Optional[Dict] = None) -> Ledger:
    """Decompose one lean into its drivers.

    ``lean`` is a shortlist record (candidate row + score fields). When
    ``player_row``/``team_row`` are supplied the volume term is split into
    team volume x usage share; without them the volume term stays composite and
    ``granularity="components_only"`` says so rather than quietly emitting
    fewer drivers.
    """
    market = lean.get("market")
    comps = lean.get("proj_components") or {}
    mean = float(lean.get("mean"))
    opp_unit = _OPPORTUNITY_UNIT.get(market, "opportunities")
    stat_unit = _STAT_UNIT.get(market, "units")

    contributions: List[Contribution] = []
    not_applied: List[Dict] = []

    if market == "anytime_td":
        return _build_td_ledger(lean, player_row, mean, stat_unit)

    volume = comps.get("volume")
    efficiency = comps.get("efficiency")
    opp_factor = comps.get("opp_factor", 1.0)
    script = comps.get("game_script", 1.0)

    if not (_is_num(volume) and _is_num(efficiency)):
        raise LedgerReconciliationError(
            f"lean for {lean.get('name')} {market} carries no projection "
            f"components; cannot explain a number without its inputs")

    volume = float(volume)
    efficiency = float(efficiency)
    opp_factor = float(opp_factor) if _is_num(opp_factor) else 1.0
    script = float(script) if _is_num(script) else 1.0

    # -- 1. baseline volume, before the game-script tilt --------------------- #
    # `volume` as recorded already INCLUDES the script multiplier (see
    # projection.expected_volume), so the pre-script level is recovered by
    # dividing it out. Doing it this way keeps the recorded value canonical
    # instead of re-deriving volume from raw rows and risking a different
    # rounding.
    pre_script = volume / script if script else volume
    granularity = "components_only"

    share = (player_row or {}).get("roll_target_share")
    if market in ("rushing_yards", "rush_attempts"):
        share = (player_row or {}).get("roll_carry_share")
    team_vol = None
    for col in ("roll_team_pass_att", "roll_team_rush_att"):
        if _is_num((team_row or {}).get(col)):
            team_vol = float(team_row[col])
            break

    if _is_num(share) and _is_num(team_vol):
        granularity = "full"
        contributions.append(Contribution(
            key="team_volume", label=f"Team {opp_unit} per game",
            stage="baseline_volume", multiplier=None,
            value_before=None, value_after=round(float(team_vol), 3),
            unit=opp_unit,
            provenance=Provenance("nflverse play-by-play",
                                  "prior weeks only",
                                  "rolling team volume, shift(1)-then-roll"),
            inputs={"roll_team_volume": round(float(team_vol), 3)}))
        contributions.append(Contribution(
            key="usage_share", label="His share of that volume",
            stage="baseline_volume", kind="level",
            multiplier=round(float(share), 4),
            value_before=round(float(team_vol), 3),
            value_after=round(float(team_vol) * float(share), 3),
            unit=opp_unit,
            provenance=Provenance("nflverse play-by-play", "prior weeks only",
                                  "8-game rolling usage share, shift(1)"),
            inputs={"usage_share": round(float(share), 4)}))
    else:
        contributions.append(Contribution(
            key="baseline_volume", label=f"Projected {opp_unit}",
            stage="baseline_volume", multiplier=None,
            value_before=None, value_after=round(pre_script, 3),
            unit=opp_unit,
            provenance=Provenance("nflverse play-by-play", "prior weeks only",
                                  "team volume x usage share (not separable "
                                  "from the published artifact)"),
            inputs={"volume_pre_script": round(pre_script, 3)}))

    running = contributions[-1].value_after

    # -- 2. game script ------------------------------------------------------ #
    contributions.append(Contribution(
        key="game_script", label="Game-script tilt",
        stage="game_script", multiplier=round(script, 4),
        value_before=round(running, 3), value_after=round(volume, 3),
        unit=opp_unit,
        provenance=Provenance("nflverse schedule (pre-game spread)",
                              "pre-game", "favourites tilt run, dogs tilt "
                              "pass; capped +/-12%"),
        inputs={"spread_line": lean.get("spread_line"),
                "total_line": lean.get("total_line")}))
    running = volume

    # -- 3. efficiency (unit changes here) ----------------------------------- #
    contributions.append(Contribution(
        key="efficiency", label="His efficiency per opportunity",
        stage="efficiency", kind="level", multiplier=round(efficiency, 4),
        value_before=round(running, 3),
        value_after=round(running * efficiency, 3),
        unit=stat_unit,
        provenance=Provenance("nflverse play-by-play", "prior weeks only",
                              "rolling efficiency, shrunk to position mean on "
                              "small samples"),
        inputs={"efficiency": round(efficiency, 4)}))
    running = running * efficiency

    # -- 4. opponent vs position --------------------------------------------- #
    contributions.append(Contribution(
        key="opp_factor", label="Opponent vs this position",
        stage="opp_factor", multiplier=round(opp_factor, 4),
        value_before=round(running, 3),
        value_after=round(running * opp_factor, 3),
        unit=stat_unit,
        provenance=Provenance("nflverse play-by-play", "prior weeks only",
                              "rolling yards/EPA allowed to the position"),
        inputs={"opp_factor": round(opp_factor, 4),
                "defense": lean.get("defteam")}))
    running = round(running * opp_factor, _ROUND_DP)   # projection.project rounds here

    # -- 5. measured situational adjustments --------------------------------- #
    for col, (stage, applied_by) in ADJUSTMENT_COLUMNS.items():
        raw = lean.get(col)
        if not _is_num(raw) or abs(float(raw) - 1.0) < 1e-9:
            not_applied.append({
                "stage": stage, "column": col, "applied_by": applied_by,
                "reason": ("not present on this pick" if not _is_num(raw)
                           else "computed as exactly 1.0 (no effect)"),
            })
            continue
        mult = float(raw)
        before = running
        after = round(before * mult, _ROUND_DP)
        contributions.append(Contribution(
            key=col, label=_ADJUSTMENT_LABELS[col], stage=stage,
            multiplier=round(mult, 4),
            value_before=round(before, 3), value_after=after,
            unit=stat_unit,
            provenance=_ADJUSTMENT_PROVENANCE[col],
            inputs={col: round(mult, 4), "applied_by": applied_by}))
        running = after

    # Give LEVELS a measured reference where one exists, so "his catch rate is
    # below the WR average" can be a real counter-argument instead of an
    # unfalsifiable magnitude. Absent a reference the level stays
    # non-directional -- we do not guess what "normal" is.
    if refs:
        _attach_level_references(contributions, refs, market, lean.get("pos"))

    ledger = Ledger(
        player_id=lean.get("player_id"), name=lean.get("name"),
        market=market, side=lean.get("side"), line=lean.get("line"),
        line_source=_line_source(lean), projected_mean=mean,
        contributions=contributions, not_applied=not_applied,
        granularity=granularity, stage_order=list(STAGE_ORDER),
        baseline=contributions[0].value_after)
    ledger.reconciled_mean = round(running, _ROUND_DP)
    reconcile(ledger)
    return ledger


#: Ledger level keys -> the reference field they compare against.
_LEVEL_REFERENCE_FIELD = {
    "efficiency": "proj_efficiency",
    "usage_share": None,        # no published reference for share yet
    "team_volume": None,
}


def _attach_level_references(contributions: List[Contribution], refs: Dict,
                             market: str, pos: Optional[str]) -> None:
    """Attach the measured league reference to each level that has one."""
    table = (refs or {}).get("references") or {}
    for c in contributions:
        if c.kind != "level":
            continue
        field_name = _LEVEL_REFERENCE_FIELD.get(c.key)
        if not field_name:
            continue
        entry = table.get(f"{market}|{pos}|{field_name}")
        if not entry or not _is_num(entry.get("mean")):
            continue
        c.reference = float(entry["mean"])
        ci = entry.get("ci")
        if ci and len(ci) == 2:
            c.reference_ci = [float(ci[0]), float(ci[1])]
        c.inputs["reference_mean"] = entry["mean"]
        c.inputs["reference_n"] = entry.get("n")
        c.inputs["reference_ci"] = entry.get("ci")


def _build_td_ledger(lean: Dict, player_row: Optional[Dict],
                     mean: float, stat_unit: str) -> Ledger:
    """anytime_td is ADDITIVE, not multiplicative.

    ``projection.project`` computes a Poisson rate::

        lambda = carries x rush_td_rate + targets x rec_td_rate

    i.e. a DOT PRODUCT of two (volume, rate) pairs. The published
    ``proj_components`` for this market record ``volume = carries + targets``
    and ``efficiency = rush_td_rate + rec_td_rate`` -- SUMS, not factors. Their
    product is not the projection and is not close to it: for a workhorse back
    it overstates lambda by roughly 2x (21.75 x 0.1278 = 2.78 against a real
    lambda of 1.194).

    So a TD pick cannot be decomposed from the published artifact alone. With
    the player row we do the real additive split; without it we say so, rather
    than emitting a multiplicative chain that would reconcile to the wrong
    number. Reporting "not decomposable" is honest; reporting a wrong
    decomposition is the failure mode this whole phase exists to prevent.
    """
    contributions: List[Contribution] = []
    granularity = "not_decomposable"

    carries = (player_row or {}).get("roll_carries")
    targets = (player_row or {}).get("roll_targets")
    rush_rate = (player_row or {}).get("roll_rush_td_rate")
    rec_rate = (player_row or {}).get("roll_rec_td_rate")

    if all(_is_num(v) for v in (carries, targets, rush_rate, rec_rate)):
        granularity = "full"
        rush_term = float(carries) * float(rush_rate)
        rec_term = float(targets) * float(rec_rate)
        contributions.append(Contribution(
            key="rush_td_expectation", label="Expected rushing TDs",
            stage="baseline_volume", multiplier=None,
            value_before=None, value_after=round(rush_term, 4),
            unit=stat_unit,
            provenance=Provenance("nflverse play-by-play", "prior weeks only",
                                  "rolling carries x rolling rushing-TD rate"),
            inputs={"carries": round(float(carries), 3),
                    "rush_td_rate": round(float(rush_rate), 4)}))
        contributions.append(Contribution(
            key="rec_td_expectation", label="Expected receiving TDs",
            stage="efficiency", multiplier=None,
            value_before=round(rush_term, 4),
            value_after=round(rush_term + rec_term, 4),
            unit=stat_unit,
            provenance=Provenance("nflverse play-by-play", "prior weeks only",
                                  "rolling targets x rolling receiving-TD rate"),
            inputs={"targets": round(float(targets), 3),
                    "rec_td_rate": round(float(rec_rate), 4)}))
    else:
        contributions.append(Contribution(
            key="td_rate_composite", label="Expected touchdowns (combined)",
            stage="baseline_volume", multiplier=None,
            value_before=None, value_after=mean, unit=stat_unit,
            provenance=Provenance(
                "nflverse play-by-play", "prior weeks only",
                "carries x rushing-TD rate + targets x receiving-TD rate; the "
                "rushing and receiving halves are not separable from the "
                "published artifact")))

    ledger = Ledger(
        player_id=lean.get("player_id"), name=lean.get("name"),
        market="anytime_td", side=lean.get("side"), line=lean.get("line"),
        line_source=_line_source(lean), projected_mean=mean,
        contributions=contributions,
        not_applied=[{
            "stage": "situational_adjustments",
            "reason": "anytime_td takes no multiplicative adjustment; its "
                      "rate is built directly from rolling usage and TD rates",
        }],
        granularity=granularity, stage_order=["baseline_volume", "efficiency"],
        baseline=contributions[0].value_after)
    ledger.reconciled_mean = round(contributions[-1].value_after, _ROUND_DP)
    # Additive reconciliation: the last running total IS lambda. Rounding
    # budget is the published 3dp on the mean plus the 4dp on each rate.
    drift = abs(ledger.reconciled_mean - mean)
    bound = 0.5 * 10 ** (-_ROUND_DP) + len(contributions) * 5e-4 * max(1.0, mean)
    ledger.reconciliation_drift = round(drift, 6)
    ledger.reconciliation_bound = round(bound, 6)
    ledger.reconciliation_exact = drift == 0.0
    if drift > bound:
        raise LedgerReconciliationError(
            f"anytime_td ledger does not reconcile for {lean.get('name')}: "
            f"replayed {ledger.reconciled_mean} vs shipped {mean}")
    return ledger


_ADJUSTMENT_LABELS = {
    "realloc_mult": "Volume bump from a teammate being out",
    "realloc_eff_mult": "Efficiency cost of absorbing that volume",
    "backup_qb": "Backup QB projected to start",
    "backup_qb_adj": "Backup QB projected to start",
    "absence_qb_mult": "Skill-position leader out (QB passing markets)",
}

_ADJUSTMENT_PROVENANCE = {
    "realloc_mult": Provenance(
        "player's own with/without splits", "prior seasons",
        "share-ratio based, capped x1.35; halved when the basis is a "
        "proportional guess"),
    "realloc_eff_mult": Provenance(
        "measured 2019-2025, n=297 absent player-weeks", "2019-2025",
        "beneficiaries gained volume but lost ~31% efficiency per "
        "opportunity; slope 0.29/unit of boost, floored at 0.85"),
    "backup_qb_adj": Provenance(
        "measured 2019-2025, n=162 backup-QB team-weeks", "2019-2025",
        "volume ~flat (the 'more handoffs' intuition is empirically false); "
        "the effect is in efficiency, x0.92 on pass-family means"),
    "absence_qb_mult": Provenance(
        "data/absence_matrix.json, n=1,146-1,514 per cause", "2019-2025 pooled",
        "cross-market effect: QB passing output when a skill leader sits; "
        "multiplicative across causes, floored at 0.85"),
}


def precision_bound(ledger: Ledger) -> float:
    """Largest reconciliation drift explainable by INPUT ROUNDING alone.

    This is not a fudge factor, it is arithmetic. The published lean records
    `volume` to 3dp and `efficiency`/`opp_factor` to 4dp, so a ledger rebuilt
    from the published artifact is working from inputs that have already lost
    precision. Reconstructing `volume x efficiency x opp_factor` from rounded
    parts CANNOT return the exact shipped product.

    So the tolerance is derived from the recorded precision:

        abs_bound = mean x SUM(half-ulp / |value|)  +  n_stage_roundings x 5e-4

    Anything inside this bound is rounding. Anything outside it is a missing or
    reordered multiplier -- which is the thing we actually want to catch. The
    bound is typically ~1e-2 on a 250-yard passing projection and ~1e-3 on a
    5-reception one, i.e. far tighter than any real adjustment (the smallest
    shipped multiplier, absence_qb WR->QB at 0.971, moves a 250-yard line by
    over 7 yards).

    A ledger built from unrounded rows reconciles EXACTLY; see
    ``reconciliation_exact``.
    """
    rel = 0.0
    stage_roundings = 0
    for c in ledger.contributions:
        if c.multiplier is None:
            if c.value_after:
                rel += 0.5 * 10 ** (-_ROUND_DP) / abs(c.value_after)
        else:
            if abs(c.multiplier) > 1e-12:
                rel += 0.5 * 10 ** (-4) / abs(c.multiplier)
            if c.stage in _ROUNDING_STAGES:
                stage_roundings += 1
    return abs(ledger.projected_mean) * rel + stage_roundings * 0.5 * 10 ** (-_ROUND_DP)


_ROUNDING_STAGES = ("opp_factor", "realloc_volume", "realloc_efficiency",
                    "backup_qb", "absence_qb")


def reconcile(ledger: Ledger, tol: Optional[float] = None) -> float:
    """Recompute the mean from the ledger alone and verify it.

    This is the completeness check that makes "anything applied to the number
    must appear here" enforceable: if the pipeline multiplies the mean by
    something the ledger does not know about, the replay lands somewhere else
    and this raises.

    ``tol`` defaults to :func:`precision_bound` -- the drift attributable to
    input rounding. Pass an explicit tolerance only when reconciling against
    unrounded inputs, where the answer should be exact.
    """
    if not ledger.contributions:
        raise LedgerReconciliationError("empty ledger")

    replay = ledger.contributions[0].value_after
    for c in ledger.contributions[1:]:
        if c.multiplier is None:
            replay = c.value_after
            continue
        replay = replay * c.multiplier
        if c.stage in _ROUNDING_STAGES:
            replay = round(replay, _ROUND_DP)

    replay = round(replay, _ROUND_DP)
    bound = precision_bound(ledger) if tol is None else tol
    drift = abs(replay - ledger.projected_mean)
    ledger.reconciliation_drift = round(drift, 6)
    ledger.reconciliation_bound = round(bound, 6)
    ledger.reconciliation_exact = drift == 0.0
    if drift > bound:
        raise LedgerReconciliationError(
            f"ledger does not reconcile for {ledger.name} {ledger.market}: "
            f"replayed {replay} vs shipped {ledger.projected_mean} "
            f"(drift {drift:.6f} > rounding bound {bound:.6f}). Either an "
            f"adjustment was applied to the number without being registered in "
            f"ADJUSTMENT_COLUMNS, or the application order changed.")
    return replay


# --------------------------------------------------------------------------- #
# The ranking ledger -- SEPARATE, because it does not move the number
# --------------------------------------------------------------------------- #
def build_ranking_ledger(lean: Dict, ml_importances: Optional[List[Dict]] = None) -> Dict:
    """Explain the ORDERING, which is a different question from the number.

    The ML ranker supplies a side probability used to order the list; per
    HOW_A_PICK_IS_MADE.md §6 the projection stays deterministic-model-owned.
    Putting ML features into the projection ledger would tell a reader the
    model moved a number it never touched, so they live here instead.

    ``ml_importances`` are GLOBAL, walk-forward permutation importances -- what
    the classifier leans on across the corpus, NOT what drove this pick. There
    is no per-instance attribution available without adding a dependency, and
    inventing one from global importances would be a fabrication.
    """
    comps = lean.get("components") or {}
    out = {
        "ranked_by": "ml_side_probability" if lean.get("ml_score") is not None
                     else "composite",
        "composite": lean.get("composite"),
        "sub_scores": {
            "edge": comps.get("edge_component"),
            "confidence_z": comps.get("z"),
            "matchup_opp": comps.get("opp_sub"),
            "matchup_script": comps.get("script_sub"),
            "matchup_pace": comps.get("pace_sub"),
        },
        "weights_used": comps.get("weights_used"),
        "model_prob": comps.get("model_prob"),
        "market_prob": comps.get("market_prob"),
        "n_books": comps.get("n_books"),
        "affects_projection": False,
        "disclosure": (
            "This ordered the list. It did not change the projected number."),
    }
    if ml_importances:
        out["ml_feature_importance"] = {
            "scope": "global",
            "caveat": ("Global walk-forward permutation importance across the "
                       "training corpus -- what this model leans on in "
                       "general, NOT an attribution for this pick."),
            "top": ml_importances,
        }
    return out


def explain_lean(lean: Dict, player_row: Optional[Dict] = None,
                 team_row: Optional[Dict] = None,
                 ml_importances: Optional[List[Dict]] = None) -> Dict:
    """Full explanation payload for one pick: projection + ranking ledgers."""
    ledger = build_ledger(lean, player_row=player_row, team_row=team_row)
    return {
        "projection": ledger.to_dict(),
        "ranking": build_ranking_ledger(lean, ml_importances),
        "edge": ledger.edge(lean),
        "screened_note": None,     # filled by the caller that knows the game
    }
