# How professional lines are made — dissection + iterative sim improvements

**Data:** 1,123 games walk-forward (eval 2020–2023, train strictly prior), closing spreads from `backtest_games.json`. Engine: `analysis/line_engine.py` + `line_engine_it6.py`; raw numbers in `book/line_engine_iterations.json`.

## Anatomy of a professional NFL line (what originators actually do)

1. **Power ratings → median margin.** Market-making books start from numerical ratings (EPA-based, opponent-adjusted, recency-weighted) plus a simulation — your `build_ratings.py` + `montecarlo.py` is architecturally the same thing.
2. **The QB carve-out.** The one player priced individually; a backup start moves openers 3–10 points. Your game sim currently makes NO QB adjustment.
3. **Situational overlays.** HFA (shrunk from ~2.8 to ~1.5, varies by team), rest differential, short weeks, travel/timezones.
4. **Key numbers.** Margins mass at 3 and 7; pros think in *cover distributions* around key numbers, not means. (Your MC already produces distributions — the missing piece is grading EV against an empirical margin kernel.)
5. **Anchor and adjust.** Openers copy consensus; the close is sharpened by order flow. The market's information set ⊇ any public model's. Corollary: **you don't beat the close by re-deriving it — only the residual it hasn't absorbed is monetizable.**

## The iterations (each ingredient measured)

| Iteration | Margin MAE | Corr | SU acc | ATS all | Kernel-EV selective |
|---|---|---|---|---|---|
| it0 points-Elo (current ratings) | 10.21 | .370 | 63.9% | **48.6%** | 48.7%, −53u |
| it1 EPA pass/rush EWM ratings | 10.88 | .121 | 56.7% | 53.4% | 52.9%, +9u |
| it2 + QB backup detection | 10.77 | .152 | 58.5% | 52.8% | 52.5%, +2u |
| it3 + rest/short-week/TZ/HFA-fit | 10.80 | .156 | 58.8% | 53.0% | **53.2%, +14.6u / 896 bets** |
| it5 market blend (α fit ≈ 0.1–0.2) | **9.81** | **.428** | **65.8%** | 52.1% | n=104 only |
| it6 combined ridge (Elo + residual) | 10.21 | .375 | 65.2% | 49.8% | 49.0%, −49u |

Disagreement-band ATS for it3 (model vs close): 1–2 pts 54.8%, **2–3 pts 55.9%**, 3–5 pts 54.3%, <1 pt 47.0%, >5 pts 52.6% — the classic sweet spot: small disagreement is noise, huge disagreement means your model is missing news.

## What the iterations prove

- **it0 vs it1/it3 is the whole lesson.** The points-Elo edge predicts *games* well (corr .37) and *spreads* not at all (48.6%): everything it knows is in the price. The EPA/QB/rest residual predicts games poorly (corr .16) and spreads best (53.0–53.4%): it's partially orthogonal to the close.
- **it6 is the cautionary tale:** merge the priced signal back in and the model collapses onto the market (ATS → 49.8%). Keep fair-value and residual models SEPARATE.
- **it5 is what your `backtest.py` verdict already said**, now quantified: blending ~85% market + ~15% model gives the best margin forecast (MAE 9.81, SU 65.8%) — that's the *fair-value* engine for CLV and line-shopping, not a side-picker.
- The +14.6u on 896 selective bets ≈ +1.6% ROI — real but thin; at −110 it needs the kernel-EV gate and the 1–5 pt band to exist at all.

## Integration plan (iterative, next passes)

1. **Ship the residual model as a game factor**: `g_market_residual` = it3 spec (EPA pass/rush edge, backup-QB flag, rest/TZ) fit against *cover* outcomes, feeding `factors.py` game features — the learning loop then weights it from graded results. (Data: all cached; ~1 day.)
2. **Key-number kernel into `backtest.py`/live EV**: replace the normal-ish sim margin tail with the empirical integer error kernel when computing cover EV (mass at 3/7 changes half-point values materially).
3. **QB adjustment into `build_ratings.py`**: backup-start detection (schedules `qb_id` vs trailing modal) → rating haircut fit walk-forward (~2–4 pts); improves fair value AND the sim's totals/props game context.
4. **Better opponent adjustment (it1 was crude)**: ridge team-strength regression per week instead of mean-centering; expect corr ~.30 while staying partially orthogonal — re-measure bands.
5. **Totals engine** same treatment (pace/PROE/wind/roof vs `total_line`) — untouched this pass.
6. Every promotion gates through `killcheck.py` + CLV, same as the prop tags.

*Caveats: 2020–2023 closes only (1,123 games); +1.6% ROI is within one bad season's variance; all grading at closing numbers, no line shopping, -110 flat. The honest posture stays: fair-value engine for price context, residual model for selective leans, CLV as referee.*
