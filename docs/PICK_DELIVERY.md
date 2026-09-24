# Pick delivery and grading runbook

`scripts/pick_delivery.py` covers the path from "the run produced cards" to "grade what
the user was actually told". It never sends messages and never refits a model. The user
keeps no books: the assistant records deliveries itself.

## 1. Prepare (generated only, nothing is delivered yet)

```
python scripts/pick_delivery.py prepare --cards reports/pick_cards_latest.json \
    --season 2026 --week 3 --kickoffs kickoffs.json --out <dir>
```

- `kickoffs.json` is `{game_id: zoned ISO kickoff}`, taken from an official schedule
  capture, never from an old brief.
- Output:
  - `manifest.json` (`status: generated_only_not_delivered`): each deliverable item with
    its exact text, sha256 and card, plus every withheld card with its reason.
  - `message.txt`: the exact text to send.
- Which cards become picks:
  - `actionable` -> PICK.
  - `watch` -> "WATCH ONLY (not a recommendation)".
  - Everything else is "no recommendation": research, pass, no live price, availability
    or inactives not established, a quote older than 6 h at preparation, no official
    kickoff, or a kickoff that has passed.
- `pick_cards.VALIDATED_MARKETS` is empty today, so no card is actionable and prepare
  can produce watch items at most.

## 2. Record (only after a real message was sent)

Record with one of two kinds of evidence:

```
# a platform-issued message id, as the platform reported it
python scripts/pick_delivery.py record --db <ledger db> --manifest <dir>/manifest.json --item <item_id> \
    --platform-message-id <id> --channel <channel> --delivered-at <zoned clock>
# or the local Hermes message row that carried the text (read-only; the row's own clock is used)
python scripts/pick_delivery.py record --db <ledger db> --manifest <dir>/manifest.json --item <item_id> \
    --hermes-message <session_id>:<messages.id> --hermes-state-db <profile>/state.db
```

- Without evidence nothing is recorded.
- A Hermes row must be an assistant message containing the item's exact text. It is
  recorded as `hermes-local:<source>:<session>:<row>`, not as a platform id. Hermes
  stores a `platform_message_id` column, but it was empty for every local assistant
  message on 2026-09-24.
- Duplicates: recording the same message twice is a no-op. Re-sending the same text adds
  a second delivered event to the same record.
- A changed pick (line, price or side) comes from a new manifest and becomes revision + 1;
  the grader keeps both given picks.
- A delivery at or after kickoff is refused unless `--retrospective` is passed, and it is
  then graded only in the retrospective section.

## 3. Thursday grading (after the final, from saved boxes)

For 2026-09-24 the target is `2026_03_ATL_GB`, ESPN event 401872948, kickoff
2026-09-25T00:15Z (20:15 EDT).

```
mkdir -p <boxdir> && curl -sf "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event=401872948" \
    -o <boxdir>/401872948.json && date -u +%Y-%m-%dT%H:%M:%SZ     # note this capture clock
python scripts/pick_delivery.py grade --db <ledger db> --season 2026 --week 3 \
    --box-dir <boxdir> --box-captured-at <capture clock> --out <grade dir>
```

- The box is rejected unless it is `STATUS_FINAL`, captured after kickoff and the only
  file for its game.
- `grade_receipt.json` pins the argv, the db and box sha256s and the counts.
- UNRESOLVED rows are listed as pending settlement and are never voided automatically.
  Sportsbook rules (inactive/one-snap action, stat-correction resettlement, push rules)
  are unverified and listed with every grade.
- Thursday rows carry `slate_tag=thursday`. A single game makes no statistical claim.
- To regrade after a stat correction, capture again and pass `--prior` to
  `scripts/grade_issued_picks.py`. Corrections are reported beside the originals.

## 4. Sunday readiness checklist

1. Run the scheduled pipeline for the slate and confirm the run published (`publish=True`
   in its receipt) and that cards were written.
2. Save an official kickoff map for the Sunday games.
3. Run `prepare` close to kickoff: quotes must be under 6 h old. If it withholds
   everything, the message is the "No pick" text and nothing is recorded.
4. Send `message.txt` unchanged, then `record` each sent item with real evidence before
   kickoff. The 6 h limit also runs to delivery: `record` refuses a pregame delivery made
   more than 6 h after the card's quote capture (even with `--retrospective`); prepare again.
5. After the finals, capture the boxes and run `grade`.
6. Keep the model separate:
   - Grades do not feed model fitting. The grader writes no `model_adjustments`.
   - Nothing retunes Sunday from Thursday's result.
   - Any refit is a separate, pre-registered step.
