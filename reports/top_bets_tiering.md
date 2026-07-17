# Accuracy-tiered per-game bet ranking (Top Bets)

Product rule: per game, "Best Bets" = up to top 5, ranked list capped at top 10.
Rank is driven by measured accuracy, fail-closed:
- BEST tier: rank ≤5 AND the pick's confidence band has graded accuracy ≥67%.
- VALUE tier: rank ≤10 AND band accuracy >50% AND edge >0.
- Bands with n<20 are "unproven" and excluded from both tiers.
- Games show FEWER bets when bands do not qualify; thresholds never relax.

## Measured bands (settled graded picks, data/weekly.json replay, 2025 season)

| Band | Accuracy | n | Tier eligibility |
|---|---:|---:|---|
| p>=0.7 (moneyline) | 69.2% | 52 | BEST |
| p 0.62-0.7 (moneyline) | 70.7% | 75 | BEST |
| p 0.55-0.62 | 51.3% | 76 | VALUE |
| p 0.5-0.55 | 59.8% | 82 | VALUE |
| edge 0.5-1.5 (ATS/total) | 61.2% | 139 | VALUE |
| edge 0.0-0.5 | 56.9% | 65 | VALUE |
| edge 1.5-2.5 | 51.2% | 125 | VALUE |
| edge 2.5-4.0 | 40.6% | 128 | EXCLUDED |
| edge>=4.0 | 46.3% | 95 | EXCLUDED |

Key honest finding: edge magnitude is NON-monotone in accuracy — the biggest
model-vs-line disagreements (edge ≥2.5) hit only 40-46% and are excluded from
tiers entirely. High-probability moneylines are the only ≥67% band family.
On the current replay this emits 616 tiered bets over 22 weeks (127 best /
489 value), ~1.8 per game — "top 10 per game" is a ceiling, not a promise.

## Provenance and limits

- Calibration uses one graded replay season; per-band n is 52-139 and shown on
  every dashboard badge. Multi-season walk-forward band recalibration
  (2021-2024 train, 2025 verify) is the registered next step before these
  tier labels are treated as season-forward guarantees.
- Synthetic/replay hit rates are research evidence, not betting edge; the
  fail-closed exact-market gate for props is untouched. Prop leans are NOT in
  the ranked tiers until their markets carry approved graded accuracy.
- Deeper ranker evidence (pooled WF 2021-24: top-1 70.10%, top-5 66.92%;
  2025 holdout: 76.47/69.93) lives in data/accuracy_registry.json.
