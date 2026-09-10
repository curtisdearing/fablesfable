#!/usr/bin/env python3
"""Collect an evidence-linked local NFL team briefing.

Examples:
    python scripts/collect_team_intel.py --team BUF --team MIA
    python scripts/collect_team_intel.py --team PHI,DAL --hours 72
    python scripts/collect_team_intel.py --all --dry-run

The output is context-only.  This command never edits model projections.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue.sources import team_intel  # noqa: E402


def _teams(values: list[str], registry: dict, all_teams: bool) -> list[str]:
    if all_teams:
        return [str(team["abbr"]) for team in registry["teams"]]
    selected: list[str] = []
    for value in values:
        for item in value.split(","):
            abbr = item.strip().upper()
            if abbr and abbr not in selected:
                selected.append(abbr)
    if not selected:
        raise SystemExit("select at least one --team (repeat or comma-separate) or pass --all")
    return selected


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--team", action="append", default=[], help="NFL abbreviation; repeat or comma-separate")
    cli.add_argument("--all", action="store_true", help="collect all 32 teams (one or more requests per team)")
    cli.add_argument("--hours", type=float, default=96.0, help="freshness window in hours (default: 96)")
    cli.add_argument("--timeout", type=float, default=15.0, help="per-request timeout in seconds")
    cli.add_argument("--include-noise", action="store_true", help="retain articles without a signal label")
    cli.add_argument("--strict", action="store_true", help="fail on the first feed error")
    cli.add_argument("--dry-run", action="store_true", help="print planned feed requests; do not fetch")
    cli.add_argument("--registry", type=Path, default=ROOT / "config" / "team_sources.json")
    cli.add_argument("--output", type=Path, default=ROOT / "data" / "team_intel_latest.json")
    cli.add_argument("--markdown", type=Path, default=ROOT / "reports" / "team_intel_latest.md")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    registry = team_intel.load_registry(args.registry)
    selected = _teams(args.team, registry, args.all)
    requests = team_intel.build_requests(registry, selected)
    if args.dry_run:
        print(json.dumps([
            {"id": row["id"], "team": row["team"]["abbr"], "method": row["method"], "url": row["url"]}
            for row in requests
        ], indent=2))
        return 0

    packet = team_intel.collect(
        registry,
        selected,
        hours=args.hours,
        include_noise=args.include_noise,
        timeout=args.timeout,
        strict=args.strict,
    )
    team_intel.write_packet(packet, args.output, args.markdown)
    quality = packet["quality"]
    print(
        f'wrote {len(packet["items"])} signal(s) for {len(packet["teams"])} team(s); '
        f'{quality["successful_requests"]}/{quality["requests"]} feed request(s) succeeded'
    )
    print(f"json: {args.output}")
    print(f"brief: {args.markdown}")
    return 0 if quality["successful_requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
