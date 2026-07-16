# Tailstail / fablesfable projection boundary

Both products may use the same football evidence, but neither product is the
other's dependency. The exchange format is schema version 1 of
`PlayerProjectionSnapshot` plus its correlated component-sample artifact.

## Shared

- stable player/game identity;
- source and information-as-of provenance;
- availability state;
- passing, rushing, and receiving event-component samples;
- canonical content hashes and Parquet integrity hashes; and
- model/simulation version metadata; and
- explicit component-validation status (`approved`, `research_only`, or
  `unvalidated`) with optional market overrides.

## Excluded from the shared contract

- fantasy scoring and league settings;
- fantasy rosters, lineup, waiver, and trade decisions;
- sportsbook lines, prices, books, vig, edge, and CLV; and
- either consumer's dashboard or release state.

The JSON summary is suitable for inspection and light consumers. The Parquet
sample artifact preserves the same simulation row across all players and
components, so game and teammate correlations survive rescoring.

Tailstail applies league scoring to the component samples. Fablesfable maps the
same component columns to stat markets, but refuses to emit a betting
probability unless the producer marks that market `approved`. Integrity is not
accuracy: a canonical hash proves which draws arrived, not that the draws beat
history. A consumer-specific field appearing in the shared snapshot fails
validation.

The contract is frozen by:

- `schemas/player_projection_snapshot.schema.json`;
- `nflvalue/projection_snapshot.py`; and
- the producer and consumer projection-contract tests.

Neither repository should track the other repository's default branch. Once
the contract has survived prospective use, its implementation can move to a
small tagged `nflvalue-core` release and both consumers can pin that version.
