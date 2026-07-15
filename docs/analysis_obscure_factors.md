# Cross-position cascades, obscure factors, and the player condition book

**Date:** 2026-07-14 · Round 3. Single factors only (no combinations yet, per instruction). Same EB shrinkage (k=60 battery, k=25 book), same synthetic-line grading (over = actual > player's trailing mean, base over-rate ≈ 43–50%). 41 new patterns tested → expect ~2 false positives at P>0.95; trust P≈1.0 + n>90 most.

## 1. Cross-position absence cascades ("seemingly unrelated" combos)

The theme: **volume redistribution is priced in; TD/red-zone redistribution is not.**

| Cascade | n | Rate vs base | Lift | P(real) |
|---|---|---|---|---|
| **TE1 out → RB1 anytime TD** | 93 | 45.2% vs 27.7% | **+10.6pp** | 0.998 |
| **WR1 out → TE1 anytime TD** | 111 | 32.4% vs 18.5% | **+9.0pp** | 0.998 |
| RB2 out → RB1 TD | 51 | 47.1% vs 27.8% | +8.8pp | 0.977 |
| WR1 out → RB1 TD | 96 | 39.6% vs 27.8% | +7.3pp | 0.975 |
| RB1 out → RB2 TD | 65 | 41.5% vs 27.8% | +7.1pp | 0.957 |
| RB2 out → RB1 carries OVER (committee collapse) | 51 | 60.8% vs 45.3% | +7.1pp | 0.933 |
| Your example, TE1 out → **RB2** TD | 69 | 17.4% vs 28.1% | **−5.7pp** | 0.065 |

Your instinct was right, your target was one seat off: when TE1 goes out, the red-zone work consolidates to **RB1** (and when WR1 is out, to TE1). RB2 actually gets *less* valuable. Red-zone touches flow to the highest-trust remaining option, not down the depth chart. Combined with round 2's RB1-out→backup-RB-carries (+15.4pp), absences are your richest unexploited factor family — and `pipeline_weekly.py` currently routes absence adjustments only to QB passing markets.

Nulls: OL 2+ out (all directions weak), opponent DB/front-7 outs ≈ 0 (books price defensive injuries), WR1 out → RB1 receptions 0.0pp.

## 2. Obscure single factors — battery results

| Factor | n | Lift | P(real) | Comment |
|---|---|---|---|---|
| **WR 100+ yds last week → rec yds UNDER** | 733 | **−13.9pp** (29.2% over) | ~1.000 | Spotlight regression: trailing-mean lines inflate after spikes and defenses adjust. Biggest single number found so far. |
| **Target spike (+5 vs trend) last wk → receptions UNDER** | 120 | −12.0pp | 0.999 | Same family. |
| RB 22+ carries last week → rush yds UNDER | 323 | −4.8pp | 0.970 | Workload hangover is real. |
| Team played OT last week → RB rush yds UNDER | 330 | −4.5pp | 0.962 | Fatigue hits RBs, not WRs (−2.8pp, P=0.91). |
| vs top-quartile blitz defense → QB pass yds OVER | 1,775 | +1.7pp | 0.925 | Mild; blitz→RB checkdowns was ~0. |
| West-coast team, 1pm ET road kick (body clock) | 271–409 | −1.8 to −2.8pp | 0.79–0.85 | The famous narrative barely shows. Monitor, don't ship. |
| International/neutral site | 240 | +0.8pp | 0.60 | Nothing. |
| Blowout win/loss letdown, rookie wall, old-WR-in-cold, ref pace crews (prop level), OL outs | — | ≤±2pp | <0.90 | Nulls — folk wisdom that doesn't survive shrinkage. |

Ref crews matter at the **game-totals** level (round 2: Hochuli 44% shrunk over-rate) but not at the prop level — pace doesn't transfer to individual lines.

## 3. The player condition book (your "lights too bright" request)

Built: **`book/player_condition_book.parquet|csv` — 32,772 rows: 868 players × 25 conditions × 3 stat families** (primary yardage, receptions, anytime TD), each with n, raw over-rate, own baseline, shrunk rate, edge; 3,993 flagged edges (|edge|≥4pp, n≥6), 3,321 on 2024–25 rosters. Plus **`book/stadium_splits.csv`** (1,156 player×venue rows, n≥4).

Sample of active-roster flags (shrunk, so these survive k=25 skepticism):

| Player | Condition | n | Raw vs base | Edge |
|---|---|---|---|---|
| C.Godwin (TB) | primetime rec yds / receptions | 13 | 8%/23% vs 48%/59% | **−13.9 / −12.3pp** — lights too bright, textbook |
| B.Aiyuk (SF) | primetime rec yds | 14 | 21% vs 57% | −12.6pp |
| D.Smith (PHI) | primetime receptions | 16 | 19% vs 52% | −13.1pp |
| A.Ekeler (WAS) | TD on turf vs grass | 33/21 | 76% vs 24% | **+11.5 / −14.5pp** — same player, surface flips his TD prop |
| M.Evans (TB) | after a 100-yd game | 12 | 8% vs 44% | −11.6pp |
| E.Elliott (DAL) | TD as big favorite | 16 | 69% vs 39% | +11.6pp |
| Z.Wilson | @ MetLife pass yds | 14 | 71% vs 46% | +10.4pp |
| E.Engram | @ TIAA Bank (own stadium!) rec yds | 15 | 27% vs 50% | −10.0pp |

Read it as a scouting index, not an auto-bet list: a 13-game primetime split is a prior, and the book's EB shrinkage already discounts it. When 2026 rosters drop, every veteran carries his history in; rookies enter the book after 3 trailing games.

## 4. In-season updating (how it keeps improving)

- **`refresh_condition_book.py`** (repo root, saved here): one command, run after Tuesday's `update_results.py`. Pulls new nflverse weeks (`NFL_SEASONS=...,2026`), rebuilds player-weeks, reruns all three batteries, rewrites `book/`. Everything is shift-then-roll, so week-N picks only ever see weeks <N.
- **Posteriors tighten automatically** — the EB machinery just accumulates n; patterns that decay get pulled toward baseline, new ones cross the credibility bar as evidence lands.
- **Roster changes**: RB1/TE1/WR2 identities are recomputed weekly from trailing usage (roster-of-record), so trades/injuries re-rank depth charts with a 1-week lag; contract-year and age refresh from nflreadpy each pull.
- **Live model integration** stays as designed in round 2: ship credible patterns as `context_features` tags (absence-TD cascades, spotlight regression, wind-pass-under, backup-QB-rush), priors = posterior logit shifts, and `prop_learning` reliability + CLV killcheck decide what survives real prices.

## 5. Answers logged from this round

`roll_pass_attempts` / `roll_carry_share` = strictly-prior EWM (span 6) of a player's attempts and share of team carries — "who owns the volume." Chemistry = currently `chemistry.py`'s `qb_chem_delta` (QB→receiver cumulative efficiency vs expectation, min 80 att) + my duo proxy (WR1 >22% target share, same modal QB 6+ games); weakly credible (+0.015). TD base: ~23% per game for rostered skill players — birthday effect was −2.2pp off that (and stays dead). Lines moving when a player's situation changes (RB2 out → higher line): correct, which is why the book grades vs trailing means and the CLV tracker remains the final arbiter of whether a pattern beats the *price*.

*Artifacts: `extended_battery.py`, `build_condition_book.py`, `refresh_condition_book.py`, `bootstrap_data.py`, `book/` (condition book, stadium splits, patterns2.json). No factor combinations were fit in this round — combination testing is the next step once you pick which singles to promote.*
