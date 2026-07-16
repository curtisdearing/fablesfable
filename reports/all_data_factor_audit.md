# Reproducible 2019–2025 factor audit

The previous hand-entered audit is retracted. Its larger denominators mixed
official pregame absences with current-game zero usage, and its beta simulation
treated the control rate as known. This run rebuilds 116,554 player-market
rows from PBP, weekly rosters, schedules, official injuries, and player
metadata. The frame's reproducibility fingerprint is a canonical CSV SHA-256,
not a Parquet-file hash: matching cells hash identically across Parquet writer
versions. Source-file and output hashes remain integrity checks. The contract
is recorded in `data/all_data_factor_audit.json`.

## What survived the corrected test

All depth labels use prior-eight-game usage among players on that week's
roster. Outcomes cannot change RB1/RB2/WR1/TE1 labels. Every line below gives
the exact exposed and control denominators; the interval samples uncertainty
in both rates, and q covers all 38 predeclared tests.

| Pregame factor | Exposed | Control | Difference | 95% posterior | BH q |
|---|---:|---:|---:|---:|---:|
| RB1 officially out → RB2 rushing yards over trailing reference | 109/155 (70.3%) | 875/2,271 (38.5%) | +31.8 pp | +24.1 to +38.9 | <0.000001 |
| RB1 officially out → RB2 anytime TD | 63/164 (38.4%) | 583/2,427 (24.0%) | +14.4 pp | +7.0 to +22.2 | 0.0010 |
| 2+ opponent DBs officially out → QB passing yards over trailing reference | 161/283 (56.9%) | 1,214/2,562 (47.4%) | +9.5 pp | +3.4 to +15.4 | 0.0197 |
| Post-bye → WR receiving yards over trailing reference | 921/2,382 (38.7%) | 4,628/11,008 (42.0%) | −3.4 pp | −5.5 to −1.2 | 0.0197 |

Observed wind of 15+ mph had a −15.7 pp QB passing result (73/216 versus
1,302/2,629; q=0.00019), but schedule weather is an observed-game proxy, not
an archived forecast. It is deliberately excluded from pregame evidence.

The leading RB1-out results are believable workload redistribution, but they
still target a player's trailing mean rather than a sportsbook number. The
line may rise when RB1 is ruled out. These are projection features worth
prospective study, not demonstrated bets.

The examples that motivated the search did not survive: TE1 out → RB2 TD was
26/132 versus 620/2,459 (−5.5 pp, q=0.43); TE1 out → RB1 TD was essentially
zero (+0.6 pp, q=0.98); birthday-window WR receiving was −2.1 pp (n=449,
q=0.60); and revenge-game WR receiving was −4.5 pp (n=211, q=0.43).

## Long-term incumbent-vacancy cohort

The official Out/Doubtful factors are intentionally a short-notice cohort:
they rank only players on the current roster. That misses the distinct case of
an established producer on IR/reserve or absent over multiple roster snapshots.
The frame builder now emits a separate research-only cohort for each QB/RB/WR/
TE role. It identifies the leading prior producer from the prior eight games
without requiring that player to be on the current roster, requires at least
three prior games and a recent team history, excludes players active for a new
team, then records reserve status or a two-week unavailability streak.

This cohort is deliberately not merged into the 38-factor published family or
the fantasy point model yet. Its protocol is: freeze the definitions before the
season; split reserve/IR, multi-week inactive, and transaction cases; test the
early replacement window separately from the established replacement window;
then require season-forward replication and calibration improvement before it
can affect a player projection. That avoids treating a cut, trade, or changing
depth chart as a generic injury boost.

## Combination projection and Monte Carlo result

Combinations of up to three predeclared factors were selected on seasons prior
to each outer test season. Training exposed/control n had to be at least 100.
No eligible procedure scored 100% or beat its segment reference:

| Segment | Outer correct/n | Accuracy | Segment reference | Selective lift | Posterior 95% |
|---|---:|---:|---:|---:|---:|
| QB1 passing yards | 93/197 | 47.2% | 50.6% | −3.4 pp | 40.3–54.2% |
| RB1 rushing yards | 84/152 | 55.3% | 56.7% | −1.4 pp | 47.4–63.2% |
| WR1 receiving yards | 164/281 | **58.4%** | 58.6% | −0.3 pp | 52.5–64.1% |

The numerical high was TE1 receiving at 66.7%, but it was only 12/18 and its
95% interval was 43.8–84.7%; it is excluded by the n≥100 rule. RB1 receptions
was likewise excluded at 31/49. The highest eligible result—WR1 receiving—was
worse than its reference, so there is no factor-combination model to promote.
The complete compact result is in `data/nested_factor_projection.json`.

## What updates in 2026

The weekly runner refreshes PBP, weekly rosters, schedules, and official
injuries while preserving prior seasons. The factor-frame builder can be rerun
after every graded week. Stadium, referee, player-specific, and conditional
chemistry interactions remain in research when no timestamped pregame source
exists; the existing prior-only chemistry features remain separate. A protocol
must be frozen before 2026 Week 1; only
prospective real sportsbook-line accuracy and CLV can promote a factor.
