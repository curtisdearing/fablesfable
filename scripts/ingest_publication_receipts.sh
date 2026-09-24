#!/usr/bin/env bash
# Record every trusted website.yml publication receipt into data/nfl_props.db (idempotent).
#
# Env: REPO (owner/name), WORK (scratch dir), GH_TOKEN for gh; optional REQUIRE_RUN (the website
# run that must be trusted, else exit 7), DB (default data/nfl_props.db) and PYTHON. Only
# successful main-branch runs of this repository's website.yml are downloaded (scripts/publication_receipt.py trusted-runs); the
# artifact is JSON that is parsed and hash-verified, never executed. Writes $WORK/summary.json.
# Exit 5 when a present receipt fails verification (the others are still recorded).
set -euo pipefail
: "${REPO:?}" "${WORK:?}"
PY="${PYTHON:-python}"
DB="${DB:-data/nfl_props.db}"
HERE="$(cd "$(dirname "$0")" && pwd)"
req=()
if [[ -n "${REQUIRE_RUN:-}" ]]; then req=(--require "$REQUIRE_RUN"); fi
mkdir -p "$WORK/receipts"
gh api "repos/$REPO/actions/workflows/website.yml/runs?status=success&branch=main&per_page=30" > "$WORK/runs.json"
ids=$("$PY" "$HERE/publication_receipt.py" trusted-runs --runs-json "$WORK/runs.json" --repo "$REPO" ${req[@]+"${req[@]}"})
for id in $ids; do
  has=$(gh api "repos/$REPO/actions/runs/$id/artifacts" \
        --jq '[.artifacts[] | select(.name == "publication-receipt" and .expired == false)] | length')
  if [[ "$has" -gt 0 ]]; then
    gh run download "$id" --repo "$REPO" -n publication-receipt -D "$WORK/receipts/$id"
  else
    echo "[ingest] website run $id: no unexpired publication-receipt (kept the live site, predates receipts, or expired)"
  fi
done
# shellcheck disable=SC2086
"$PY" "$HERE/publication_receipt.py" ingest --db "$DB" --receipts-root "$WORK/receipts" \
  --runs $ids --summary "$WORK/summary.json"
