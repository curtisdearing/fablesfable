# Factor research protocol

This directory is the reproducible replacement for ad-hoc pattern scripts.
None of its outputs alter live scores.

```bash
python -m analysis.build_factor_frame --output data/factor_frame.parquet
python -m analysis.all_data_factor_audit \
  --input data/factor_frame.parquet \
  --patterns analysis/factor_patterns.json \
  --output data/all_data_factor_audit.json
```

The frame records schedule/venue, prior-game workload, birthday, revenge,
official injury, and prior-usage depth factors for every eligible player-week.
The long-form output retains player, team, opponent, role, depth rank, market,
actual, strictly-prior trailing reference, exact eligibility, season, stadium,
and referee. This permits a player-specific condition history once rosters are
available without pretending a six-game stadium split is stable.

The predeclared family is in `factor_patterns.json`. Stadium, referee, and
player-specific categorical levels are retained in the frame but are not
individually published from the same data used to discover them. They may enter
`nested_factor_selection.py`; each category or conjunction must be selected on
earlier seasons and evaluated on a later untouched season.

Important boundaries:

- “Roll pass attempts” and “roll carry share” are prior-game rolling estimates:
  average QB attempts and the share of team rushes assigned to a player. They
  are predictors, not current-game outcomes.
- Official Out/Doubtful status is pregame. Zero current-game usage is not.
- Historical schedule temperature/wind are observed-game proxies. Their
  columns are explicitly labeled `_proxy` and cannot validate a forecast edge.
- The trailing mean outcome is a research diagnostic, not a sportsbook line.
  Promotion requires prospective real-line accuracy and CLV.
- The existing chemistry module computes strictly-prior QB/receiver and
  shotgun splits. The specific backup-QB × weather × surface, referee-pace,
  and injury-conditioned chemistry interactions are not promoted here until
  timestamped starter/weather/inactive inputs exist. Actual starters,
  formations, or final weather cannot be backfilled and called pregame evidence.
