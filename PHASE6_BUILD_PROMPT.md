<!--
HOW TO USE: run after Phases 1-5 are merged. Paste below the divider into Fable.
Source: docs/AUDIT_stats_context_2026-07-03.md — read that first, it's the "why"
behind every item below.
-->

---

# Build Prompt — Phase 6: matchup granularity, weather calibration, injury/context depth

You are implementing **Phase 6**. Phases 1–5 are complete and merged. This phase
closes gaps found in `docs/AUDIT_stats_context_2026-07-03.md`: several features are
computed but unused, one heuristic (weather) is guessed instead of measured, and a
few plausible signal families were never extracted. Post a plan and wait for approval
before building. You have latitude on exact implementation — the "what" and "why"
below are firm, the "how" is yours to design well.

## 0. Read first
- `docs/AUDIT_stats_context_2026-07-03.md` (full context for every item below).
- `nflvalue/features.py`, `projection.py`, `advanced_features.py`, `composite.py`,
  `candidates.py`, `sources/weather.py`, `factors.py`, `sources/availability.py`,
  `context_features.py`, `context_study.py`.

## 1. Constraints (all prior non-negotiables still apply)
- Leakage/walk-forward guards extend to every new feature — no exceptions.
- Any new narrative/context tag (revenge subtypes, primetime, travel, etc.) goes
  through the existing `context_study.py` gate (n≥100, BH-q corrected) before it's
  allowed to influence anything beyond the context panel.
- Weather and any other adjustment must be **fit from this project's own historical
  data**, not hand-picked thresholds — that's the whole point of this phase.
- Defender/trench-level matchup data: feasibility-check against free sources first
  (NGS, charting) — if it's not honestly buildable free, say so and stop there rather
  than approximating it and calling it real.

## 2. Scope — build in order

**6.1 Matchup granularity.** Split `build_opp_pos_def` finer than WR/TE (slot vs.
perimeter, man vs. zone where FTN charting allows); make the shrinkage prior in
`features.py` archetype-aware instead of coarse-position-only (e.g. receiving RBs vs.
early-down backs); add red-zone defense allowed as an opponent feature. Pin down and
document the composite matchup sub-score weighting in `composite.py` — no more silent
1/3-vs-1/4 shifts.

**6.2 Red-zone / TD wiring.** Reference the red-zone target/carry share
`advanced_features.py` already computes inside `projection.py`'s anytime-TD math.
Extend the RB1-out reallocation to cover goal-line/TD share, not just
targets/carries share.

**6.3 Game script.** Bring PROE and pace into the deterministic game-script tilt
alongside the spread-based cap, not just the ML layer. Add a neutral-script (garbage
time) filter to the core rolling usage/efficiency columns in `features.py`, keyed off
score differential late in games — mirror however the existing PROE/pace neutral
filter already does this.

**6.4 Weather, properly measured.** Refit `factors.py`'s severity weights and
thresholds from actual historical pbp temp/wind/precip vs. passing-yard and FG%
splits — same walk-forward-measured-constant standard as the backup-QB/absence
multipliers. Add wind direction relative to stadium orientation. Give kickers their
own weather-sensitivity treatment (FG% by distance × wind/temp), not just skill
positions. Add a dome-vs-open-roof player performance split, and a small script that
pairs each stadium's roof status with that game's weather (does a retractable roof
close when forecast is bad — worth knowing whether "dome" during the game means
weather was already neutralized by the time of kickoff).

**6.5 Injuries, deeper.** Build an opponent-side `ABSENCE_MULT` analog into the
composite matchup score, reusing the team's-own-outs infrastructure. Add O-line-out
effects on the team's own sack rate and QB scramble/rushing props. Add a durability
feature: how often a player has left games early / not finished, from injury and
snap data.

**6.6 Situational context, expanded.** Split revenge by transaction type (trade vs.
cut vs. free agent) and retest each subtype independently — weight whichever clears
the significance gate. Stratify the existing birthday/revenge tests by home/away and
opponent quality instead of testing unconditional pooled effects. Add and test:
primetime/noteworthy game flag (TNF/SNF/MNF) with player performance conditioned on
it, travel/timezone distance for the upcoming game, and player-level home/away,
weather-conditioned, and rivalry/division-matchup splits.

**→ CHECKPOINT after 6.1–6.3, again after 6.4–6.6:** show before/after backtest deltas
(accuracy + composite/ML hit rate) and any newly-significant context tags, then wait.

## 3. Out of scope
No new paid data sources. No scoring of any context tag that hasn't cleared the
existing significance gate. No bet placement.

## 4. Tests & definition of done
- Leakage tests extended to cover every new feature.
- Weather coefficients traceable to a fitted regression against historical splits,
  documented like the other measured constants in `DATA_SOURCES.md`/`decisions_p3-5.md`.
- Composite matchup weighting is fixed and documented, not conditionally silent.
- Each new feature has a walk-forward ablation showing it moves accuracy or hit rate
  (or an honest note that it didn't).
- Any new narrative tag is reported with its n, effect size, and q-value — scored
  only if it clears gate.

## 5. Protocol
Read docs → plan → 6.1–6.3 checkpoint → 6.4–6.6 checkpoint. Branch + small commits.
