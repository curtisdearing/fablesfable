# Audit: extraction quality and improvement opportunities

Date: 2026-07-03. Scope: stats/projection engine, weather, injuries/availability,
narrative context (revenge/birthdays), and matchup-mismatch scoring, reviewed
against the code in `nflvalue/` and the claims in `docs/`. Overall the
pipeline is unusually disciplined — measured constants over vibes, walk-forward
guards, honest gating of narrative tags — so this audit focuses on where that
discipline hasn't yet been applied, or where a real signal may be getting
tested in a way that hides it.

## 1. Core stats and projection engine

The rolling usage/efficiency stats in `features.py` shrink small samples
toward a league prior using a fixed pseudo-count (`SHRINK_K=6.0`) and a prior
mean based only on coarse position (QB/RB/WR/TE). That means a rookie
pass-catching back regresses toward a blend of early-down grinders and
receiving backs, and a big-slot receiver regresses toward the same prior as a
deep-perimeter burner. The opponent-defense factor (`build_opp_pos_def`) is
split WR vs. TE but goes no finer — no slot/perimeter or man/zone split,
something `decisions_p3-5.md` already flags as a queued-but-unbuilt ablation.
Tightening both — archetype-aware priors and a finer opponent split — is the
single highest-leverage fix in this section, since it touches every player
projection rather than a narrow subset of games.

Two specific blind spots stand out. First, `advanced_features.py` computes
red-zone target/carry share and internally describes it as "the anytime-TD
driver raw volume misses," but `projection.py`'s anytime-TD math never
references it — TD expectation is still blended overall rate × volume. That's
a concrete, low-effort fix for TD props specifically. Second, PROE and pace
are computed but, by the code's own documentation, only feed the ML layer —
the deterministic game-script tilt is driven purely by the pre-game spread,
capped at ±12%. That's a defensible design choice (keep the auditable path
simple), but it means the README's framing of PROE/pace as inputs to "the
tilt" overstates what the deterministic formula actually uses.

There's no garbage-time filter anywhere in the core rolling stats
(`roll_*` columns) that feed `mean = volume × efficiency` — only the separate
PROE/pace features filter for neutral situations. Blowout-inflated volume or
garbage-time efficiency can leak directly into the number that matters most.
Given the project already has the machinery (score differential is used
elsewhere), adding a neutral-script filter to the core rolling window is
probably the next-best return on effort after the shrinkage/opponent-split
fix.

The composite matchup sub-score (`composite.py`) averages opponent-yards
factor, game-script fit, and pace, but the EPA-allowed term is only
conditionally appended — so the effective weighting silently shifts between
1/3 and 1/4 depending on data availability, undocumented. Worth pinning to a
fixed weighting or explicitly documenting when it drops out.

The biggest structural gap — no defender-specific or trench-level matchup
data (shadow coverage, O-line vs. D-line) — is already acknowledged in
`decisions_p3-5.md` and `HOW_A_PICK_IS_MADE.md` as a free-data limitation
post-2023, with formation-adjacent signals substituted deliberately. That's a
reasonable trade given the constraint; it doesn't need fixing so much as
staying visible as the ceiling on "matchup mismatch" claims.

## 2. Weather

Weather is computed in two places that don't share ground truth. `factors.py`
has a hand-built severity heuristic (`0.55·wind + 0.30·precip + 0.15·cold`,
with hard thresholds like 30mph wind and 20F cold) feeding the linear factor
model; `advanced_features.py` attaches raw temp/wind (no precipitation) to
the GBDT features, with domes neutralized to 70°F/0mph. The heuristic weights
and thresholds are guessed, not derived from the historical pbp temp/wind
data the project already has — a direct exception to the "measured constants,
not vibes" standard applied everywhere else (backup-QB penalty, absence
multipliers, etc.). Fitting those coefficients from real passing-yard and
FG% splits against wind/precip would bring weather in line with the rest of
the codebase and is probably worth doing before anything else here.

Two coverage gaps are worth closing: wind direction relative to field
orientation is never pulled (only wind speed as a scalar — a crosswind and a
downfield gust are treated identically, despite this being one of the most
cited real effects for kicking and deep passing), and dome status is a static
hardcoded list, so retractable-roof stadiums (AT&T, State Farm, Allegiant,
etc.) always read as neutral even on an open-roof cold/wet game — exactly the
games where bettors are also likely to have forgotten to check roof status,
so this is a lost edge rather than just an error. Altitude (Denver) isn't
represented anywhere despite thin-air effects on kicking distance being
well-documented. None of these require new data sources — Open-Meteo already
returns wind direction, and roof-state and elevation are static facts that
just need to be entered.

## 3. Injuries and availability

This is the strongest-built section of the four: two-clock design (Wednesday
league-wide pull, T-90 roster-confirmation override), honest degradation to a
flagged "proportional_guess" for thin samples, and reasonably large, disciplined
sample sizes on the measured multipliers (n=297 teammate-out, n=162
backup-QB, n=1,146–1,514 per absence cause). The gaps are about coverage
breadth rather than rigor.

The most consequential: opponent injuries never reach the composite matchup
score. A defense missing its top corner or edge rusher is only visible
through an opaque ML-ranker feature and a context-panel note — there's no
auditable, measured multiplier the way the team's-own-outs get one. Given
this is exactly the kind of mismatch a "matchup" score should capture,
building an opponent-side `ABSENCE_MULT` analog is a natural next step and
reuses infrastructure that already exists for the team's-own-player-out case.

Two adjustment families that exist for team's-own-outs haven't been extended
to plausible neighbors: RB1-out effects on other RBs' red-zone/TD share
specifically (the current reallocation only covers targets/carries share,
not goal-line share, which matters for anytime-TD props), and O-line-out
effects on the team's own sack rate or QB scramble/rushing props — the
pressure-rate infrastructure already exists on the defensive side, just not
mirrored for the offense's own injuries.

Smaller but real: injury tags collapse DNP→Limited→Full practice trajectories
into a single final-day status, with no base-rate calibration for how often a
"Questionable" tag actually plays — a practice-trend signal that's known to
often out-predict the final tag. And the T-90 feed's own docstring calls
itself "the single riskiest link in the chain," which is honest, but worth
confirming the freshness gate is sized in minutes for that clock rather than
inheriting the 36-hour threshold built for the Wednesday clock.

## 4. Revenge games and birthdays

The measured conclusion so far — that neither clears significance
(birthday: 36.3% vs. 36.6% baseline, n=2,275; revenge: 33.7% vs. 36.7%,
n=1,111) — is probably right for birthdays as currently defined, but the way
both are operationalized could be hiding a real, narrower effect rather than
proving there isn't one.

Both tags are tested pooled across every market at once. If a birthday or
revenge effect exists, motivation-driven stories point toward touchdowns and
red-zone usage specifically, not garbage yardage — pooling receptions,
yardage, attempts, and TDs into one binomial test would wash out a
market-specific signal even if one were real. Re-running the test stratified
by market (with a market-aware multiple-comparison correction) is the
cheapest possible next step, since it uses data the pipeline already has.

Revenge is also defined too broadly to be a clean test: "rostered on the
opponent for ≥3 weeks, ever" fires identically for a player traded away in
his prime, a career backup cut in camp, and a plain free-agent move — three
scenarios with plausibly opposite emotional valence, mixed into one n=1,111
pool. Splitting by transaction type (trade vs. cut vs. free agent) before
testing is a moderate-effort fix that directly targets why the current
number reads as noise. Team-level revenge (lost the last meeting) and
coaching revenge (fired coordinator/HC facing the old team) — both more
commonly cited in betting commentary than player-level roster history —
aren't extracted at all.

Neither tag controls for home/away, spread, or opponent quality, so a true
small effect could be confound-cancelled rather than absent — for instance,
if birthday games happen to skew toward tougher primetime matchups. Adding
those as covariates or stratification checks would separate "no effect" from
"real effect, masked."

Finally, several well-established situational families with more plausible
mechanisms than birthdays — primetime/national TV, short-week Thursday games,
look-ahead/trap spots, divisional familiarity, and long-distance/timezone
travel for early West-to-East kickoffs — aren't extracted anywhere in the
codebase, let alone tested. These are more likely to actually clear
n≥100/q<0.05 than any re-slicing of birthdays, and building them is probably
higher expected value than further birthday analysis.

## Priority order across all four areas

If only a few things get built next: (1) fit weather's severity coefficients
from real historical splits instead of the current guessed thresholds — the
clearest violation of the project's own stated standard; (2) wire an
opponent-side injury multiplier into the composite matchup score, since the
infrastructure for the team's-own-outs case already exists; (3) wire red-zone
share into the anytime-TD projection, a narrow and well-scoped fix; (4)
stratify the birthday/revenge significance tests by market and revenge
subtype before concluding there's no signal there; (5) add the missing
situational families (primetime, short week, travel, trap games) as new
tested tags, since they're more plausible than what's already been ruled out.
