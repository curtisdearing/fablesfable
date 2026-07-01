"""The kill-check: after ~150 logged leans, does this thing actually work?

PROP_SHORTLISTER_SPEC.md §5 / PREMORTEM.md: if the forward leans don't beat
the market after ~150 logged props, the composite is not finding real prop
edges -- the honest response is to revert to "projection/entertainment tool"
and stop staking, not to explain the sample away. This module renders that
verdict mechanically so nobody (including future-us) can argue with it.

Verdicts:
  INSUFFICIENT_SAMPLE  n < min_sample: keep logging, draw no conclusion.
  GO                   avg CLV > 0 AND positive-CLV rate >= 52%: the leans
                       systematically beat the close -- consistent with edge.
  NO_GO                anything else at n >= min_sample: KILL CRITERION MET.

The naive baseline is built in: CLV measures our entry against the SAME
side at the SAME book-consensus close, i.e. exactly "take the model's side
at the posted number" -- a strategy with zero timing skill scores ~0 here
(minus noise), so beating 0 with a >=52% hit rate is the bar.
"""

from __future__ import annotations

from typing import Dict

from . import db as dbmod
from .clv import rolling_clv

DEFAULT_MIN_SAMPLE = 150
POSITIVE_RATE_BAR = 0.52


def report(conn=None, min_sample: int = DEFAULT_MIN_SAMPLE, window: int = 50) -> Dict:
    conn = conn or dbmod.connect()
    stats = rolling_clv(conn, window=window)
    n = stats["n"]

    leans_n = int(dbmod.query_df(
        conn, "SELECT COUNT(*) AS n FROM leans WHERE status='active'").iloc[0]["n"])

    if n < min_sample:
        verdict = "INSUFFICIENT_SAMPLE"
        detail = (f"{n} leans with resolved CLV (of {leans_n} logged; "
                  f"{min_sample} needed). Keep logging; no conclusion yet — "
                  "and no staking conclusions either way.")
    elif (stats["lifetime_mean"] or 0) > 0 and (stats["positive_rate"] or 0) >= POSITIVE_RATE_BAR:
        verdict = "GO"
        detail = (f"Avg CLV {stats['lifetime_mean']:+.4f} prob-points over {n} leans, "
                  f"{stats['positive_rate']:.0%} beat the close (bar: {POSITIVE_RATE_BAR:.0%}). "
                  "Consistent with real edge. Staking still means quarter-to-half Kelly on a "
                  "SHRUNK edge, hard per-bet cap, fixed monthly loss limit (spec §8).")
    else:
        verdict = "NO_GO"
        detail = (f"KILL CRITERION MET: avg CLV {stats['lifetime_mean']:+.4f}, positive rate "
                  f"{(stats['positive_rate'] or 0):.0%} over {n} leans — the leans do not beat "
                  "the close. Revert to projection/entertainment tool; stop staking. "
                  "(PROP_SHORTLISTER_SPEC.md §5.3 — this outcome was pre-committed.)")

    return {**stats, "leans_logged": leans_n, "min_sample": min_sample,
            "verdict": verdict, "detail": detail}


def main() -> None:  # pragma: no cover - thin CLI
    import json
    print(json.dumps(report(), indent=2))


if __name__ == "__main__":
    main()
