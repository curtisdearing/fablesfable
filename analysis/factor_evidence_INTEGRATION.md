# Factor evidence: integration recipe (for the integration lead)

Module: `nflvalue/factor_evidence.py` (pure; stdlib + `explain_render.BANNED_TERMS`; no I/O, no network).
Tests: `tests/test_factor_evidence.py`. Real-source example: `python -m analysis.factor_evidence_examples`.

This module does not edit `pick_cards.py`, `pipeline_weekly.py` or the site. The lead wires the four steps below.

## 1. Run receipt (pipeline_weekly.run_week, after step 4c)
Record what actually executed. Do not derive this from the feature list.
```python
receipt = {
  "component": football_forecast.FORECAST_VERSION,
  # add "realloc_volume"/"realloc_efficiency"/"absence_qb" only if the OUT gate evaluated
  # this run (even when zero players were OUT); add "backup_qb" when
  # apply_backup_qb_adjustment ran; "dispersion" and "game_script" always run in enumerate_candidates
  "stages_executed": [...],
  "primary_margin_source": football_forecast.PRIMARY_MARGIN_SOURCE,
  "ordering_component": "ml_ranker" if _maybe_stamp_ml actually stamped scores else None,
  "ordering_features_populated": [f for f in ml_features if cands[f].notna().any()],
}
```
Stage multipliers already on the row (`realloc_mult`, `realloc_eff_mult`, `backup_qb_adj`,
`absence_qb_mult`) are the executed values. The adapter shows them verbatim (`x1.085 on the projected mean (as executed)`).
- Stage ran and left no column → "Checked; no change".
- Stage not in the receipt → "Not verified; not used" (missing, not zero).

## 2. News/context records (per game, before cards)
```python
news = fe.assess_news(items, as_of, model_links={(player_id, "availability"): {
          "component": "availability_gate", "feature_name": "injury_status",
          "consumed": True, "reason": "status OUT removed the player"}})   # only for players the OUT gate acted on
recs = news + fe.records_from_forecast_row(row, receipt, as_of) + specialist_records
```
- `items` come from existing feeds only: `sources/availability.py` rows (tier `data_feed`), `sources/espn_news.py`, and team/league pages already fetched with a generic UA. Record `source_tier`, `claim_kind`, `attribution`, the three clocks, and `story_id` (the same story across outlets gets the same id).
- Opportunity/personnel specialists: pass their contract-shaped dicts through `fe.normalize_record`. It re-derives status from `consumed`/`consumed_shadow`/`evaluated_neutral`/`populated`, so a claimed `numeric_applied` without consumption is downgraded.
- Splits: `fe.split_context_record(...)` needs `games` (ids + dates ≤ cutoff), `split_n`, and the baseline n. Splits with n < 5 read "Insufficient evidence (n=…)".
- Schedule: `fe.schedule_context_record(team, games, as_of, sources)`. It is verified only when a team/league source agrees.

## 3. Card surface (pick_cards.render_html, one line per card)
```python
from nflvalue import factor_evidence as fe
panel = fe.build_panel(fe.select_for_card(recs, card | {"team": row_team}), as_of)
card["factor_panel"] = panel                       # JSON payload
parts.append(fe.render_panel_html(panel))           # inside the card <div>
```
`render_panel_html` escapes everything and only links `https://` URLs (rel=nofollow noopener). It emits no script or style; class names are `fe-*`.
`build_panel` raises `fe.UnsafeCopy` on imperative or certainty wording. Catch it and drop the panel; do not publish it.

## 4. Status → label (exact strings)
| status | label |
|---|---|
| numeric_applied, isolated effect | `Used in projection: x{mult} on the projected mean (as executed)` |
| numeric_applied, not isolated | `Used in projection; contribution not isolated` |
| considered_no_change | `Checked; no change to the projection` |
| shadow_only | `Tested in shadow only; not in the published projection` |
| context_only | `Context only; not used by the projection` |
| context_only + ordering_consumed | `Enters the ordering score only; does not change the projection` |
| unavailable_unverified | `Not verified; not used` (+ `Why not used: …`) |

## Known gaps the lead must not paper over
- No Week-3 run receipt exists yet. The real example's model-stage rows come from code config at d3bd27e and say so.
- The ML ranker consumes `team_margin` and `total_line`, which are market-derived. The card already calls the ordering score "market-informed". This module labels ranker features as ordering-only.
- Running `tests/` needs the pinned `historical/` inputs (including the rosters cache) and `historical_lines.parquet`. Without them, 18 roster-dependent tests fail offline. With them provisioned in a disposable tree and the network blocked, the full suite passes.
- Superseding news: a strictly later claim from an equally or more reliable tier supersedes an earlier one on the same topic. Otherwise differing claims are flagged "Sources disagree".
