"""Advisory tuner for the selector's per-market probability-edge bars.

This is the script ``nflvalue/selector.py``'s ``tuning_note`` promises:
"prints replay hit rates per bar". It reads GRADED picks out of the warehouse
and shows, per market and per tier, whether each bar (lean / playable / strong)
is actually set where the realized hit rate clears the -110 breakeven.

ADVISORY ONLY. This script NEVER edits config.json and NEVER places a bet. It
reads the DB (read-only), prints tables, and -- optionally -- writes a clearly
labelled SUGGESTION file (data/selector_tuning_suggestion.json) that a human
reviews. Nothing here is wired into the live pipeline.

Two signals, kept separate on purpose:
  * picks HIT RATE is the SYNTHETIC-LINE signal. Many graded picks were scored
    against model/synthetic lines, not a real closing number, so a "hit" here
    is the model beating its own line -- informative for ranking bars, but NOT
    proof of market edge. Treat it as directional.
  * CLV (from the ``clv`` table) is the REAL edge signal: did we beat the
    closing line? Where CLV rows exist they carry more weight than raw hits.

Breakeven at standard -110 juice is 110/210 = 0.5238; a tier has to clear that
to be worth playing. Suggestions require a minimum sample (default n>=30)
before moving any bar; below that the honest answer is "keep logging".

Deterministic and read-only. Run:
    python3 scripts/tune_selector_thresholds.py [--season 2025] [--write]

Leans, not locks. Advisory/research only; no bet placement. 1-800-GAMBLER.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import db as dbmod            # noqa: E402
from nflvalue.selector import (             # noqa: E402
    DEFAULT_SELECTOR_CFG,
    selector_config,
)

# -110 breakeven: you risk 110 to win 100, so you must win 110/210 of the time.
BREAKEVEN = 110.0 / 210.0                   # 0.52380952...
MIN_N = 30                                  # sample floor before suggesting a move
MARGIN = 0.02                               # comfort band around breakeven for a suggestion
SUGGESTION_PATH = os.path.join(ROOT, "data", "selector_tuning_suggestion.json")


def _load_config() -> dict:
    """Read the live selector config (merged over shipped defaults). Falls back
    to shipped defaults if config.json is missing/unreadable -- read-only."""
    cfg = {}
    cfg_path = os.path.join(ROOT, "config.json")
    try:
        with open(cfg_path, "r") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        cfg = {}
    return selector_config(cfg)


def _graded_picks(conn, season):
    """Graded ACTIVE picks as a DataFrame (status='active' AND hit IS NOT NULL).
    Optional season filter. Read-only."""
    sql = ("SELECT season, week, player_id, market, side, tier, edge, ev, "
           "model_prob, market_prob, hit "
           "FROM picks WHERE status='active' AND hit IS NOT NULL")
    params = ()
    if season is not None:
        sql += " AND season=?"
        params = (season,)
    return dbmod.query_df(conn, sql, params)


def _fmt_rate(n, hits):
    if n <= 0:
        return "   n/a"
    return f"{hits / n:6.3f}"


def _flag(rate, n):
    if n <= 0:
        return ""
    if rate < BREAKEVEN:
        return "  <-- BELOW breakeven"
    return ""


# -------------------------------------------------------------------------- #
# Section 2: overall + per-market/per-tier hit rates vs breakeven             #
# -------------------------------------------------------------------------- #
def print_by_tier(df):
    print("\n== BY TIER (overall, like picks_record) ==")
    print(f"   breakeven @ -110 = {BREAKEVEN:.4f}")
    print(f"   {'tier':<10}{'n':>6}{'hit_rate':>10}   flag")
    for tier, grp in df.groupby("tier"):
        n = len(grp)
        rate = float(grp["hit"].mean())
        print(f"   {tier:<10}{n:>6}{rate:>10.3f}{_flag(rate, n)}")


def print_by_market_tier(df):
    print("\n== BY MARKET x TIER ==")
    print(f"   {'market':<18}{'tier':<10}{'n':>6}{'hit_rate':>10}   flag")
    for market, mgrp in df.groupby("market"):
        for tier, grp in mgrp.groupby("tier"):
            n = len(grp)
            rate = float(grp["hit"].mean())
            print(f"   {market:<18}{tier:<10}{n:>6}{rate:>10.3f}{_flag(rate, n)}")


# -------------------------------------------------------------------------- #
# Section 3: replay hit rates per edge bin, per market (the tuning_note)      #
# -------------------------------------------------------------------------- #
def _market_bars(cfg, market):
    thr = cfg["thresholds"]
    return thr.get(market, thr["default"])


def print_edge_bins(df, cfg):
    print("\n== REPLAY HIT RATES PER EDGE BIN (per market) ==")
    print("   edge bins are drawn around the CURRENT lean/playable/strong bars;")
    print("   read down the column to see whether each bar sits where the")
    print("   realized hit rate actually clears breakeven.")
    for market, mgrp in df.groupby("market"):
        bars = _market_bars(cfg, market)
        lean, play, strong = bars["lean"], bars["playable"], bars["strong"]
        edges = [-1e9, lean, play, strong, 1e9]
        labels = [
            f"edge < lean({lean:.3f})",
            f"lean..playable [{lean:.3f},{play:.3f})",
            f"playable..strong [{play:.3f},{strong:.3f})",
            f"strong+ [>= {strong:.3f}]",
        ]
        print(f"\n   -- {market}  (lean={lean:.3f} playable={play:.3f} strong={strong:.3f}) --")
        print(f"      {'edge bin':<34}{'n':>6}{'hit_rate':>10}   flag")
        e = mgrp[mgrp["edge"].notna()]
        for i, label in enumerate(labels):
            lo, hi = edges[i], edges[i + 1]
            binned = e[(e["edge"] >= lo) & (e["edge"] < hi)]
            n = len(binned)
            if n == 0:
                print(f"      {label:<34}{0:>6}{'   n/a':>10}")
                continue
            rate = float(binned["hit"].mean())
            print(f"      {label:<34}{n:>6}{rate:>10.3f}{_flag(rate, n)}")


# -------------------------------------------------------------------------- #
# Section 4: CLV join -- the real edge signal                                #
# -------------------------------------------------------------------------- #
def print_clv_by_tier(conn, df, season):
    print("\n== CLV BY TIER (real edge signal; picks-hit is synthetic) ==")
    sql = "SELECT season, week, player_id, market, side, clv_prob FROM clv"
    params = ()
    if season is not None:
        sql += " WHERE season=?"
        params = (season,)
    clv = dbmod.query_df(conn, sql, params)
    if clv.empty:
        print("   no CLV rows for this filter -- CLV accrues as leans close;")
        print("   until then only the synthetic-line hit rate is available.")
        return
    joined = df.merge(
        clv, on=["season", "week", "player_id", "market", "side"], how="inner"
    )
    if joined.empty:
        print("   graded picks and CLV rows do not overlap for this filter yet.")
        return
    print(f"   {'tier':<10}{'n':>6}{'avg_clv':>10}{'pos_clv_rate':>14}")
    for tier, grp in joined.groupby("tier"):
        n = len(grp)
        avg = float(grp["clv_prob"].mean())
        posrate = float((grp["clv_prob"] > 0).mean())
        print(f"   {tier:<10}{n:>6}{avg:>10.4f}{posrate:>14.3f}")


# -------------------------------------------------------------------------- #
# Section 5: threshold-move suggestions (print only; optional file)          #
# -------------------------------------------------------------------------- #
def build_suggestions(df, cfg):
    """For each (market, tier) with adequate n, suggest raising the bar if the
    tier is below breakeven, or lowering it if it comfortably clears breakeven.
    Returns a list of suggestion dicts. Never mutates config."""
    suggestions = []
    # tier -> the config threshold key that gates it
    tier_bar = {"LEAN": "lean", "PLAYABLE": "playable", "STRONG": "strong"}
    for market, mgrp in df.groupby("market"):
        bars = _market_bars(cfg, market)
        for tier, grp in mgrp.groupby("tier"):
            bar_key = tier_bar.get(tier)
            if bar_key is None:            # RESEARCH / PASS have no playable bar
                continue
            n = len(grp)
            rate = float(grp["hit"].mean())
            cur = bars[bar_key]
            rec = {
                "market": market, "tier": tier, "bar_key": bar_key,
                "current_bar": round(cur, 4), "n": n,
                "hit_rate": round(rate, 4),
            }
            if n < MIN_N:
                rec["action"] = "keep_logging"
                rec["suggested_bar"] = round(cur, 4)
                rec["reason"] = f"insufficient sample (n={n} < {MIN_N})"
            elif rate < BREAKEVEN - MARGIN:
                # losing money at this bar -> demand a bigger edge
                rec["action"] = "raise_bar"
                rec["suggested_bar"] = round(cur + 0.010, 4)
                rec["reason"] = (f"hit {rate:.3f} < breakeven {BREAKEVEN:.3f} "
                                 f"with n={n}: bar is too loose")
            elif rate > BREAKEVEN + MARGIN:
                # comfortably profitable -> capture more by loosening slightly
                rec["action"] = "lower_bar"
                rec["suggested_bar"] = round(max(cur - 0.005, 0.0), 4)
                rec["reason"] = (f"hit {rate:.3f} > breakeven {BREAKEVEN:.3f} "
                                 f"with n={n}: room to capture more")
            else:
                rec["action"] = "hold"
                rec["suggested_bar"] = round(cur, 4)
                rec["reason"] = f"hit {rate:.3f} within breakeven band; leave as-is"
            suggestions.append(rec)
    return suggestions


def print_suggestions(suggestions):
    print("\n== SUGGESTED BAR MOVES (advisory only -- NOT applied) ==")
    if not suggestions:
        print("   no (market,tier) groups to evaluate.")
        return
    print(f"   {'market':<18}{'tier':<10}{'bar':<10}{'cur':>7}{'sugg':>8}"
          f"{'n':>6}{'hit':>8}   action")
    for s in suggestions:
        print(f"   {s['market']:<18}{s['tier']:<10}{s['bar_key']:<10}"
              f"{s['current_bar']:>7.3f}{s['suggested_bar']:>8.3f}"
              f"{s['n']:>6}{s['hit_rate']:>8.3f}   {s['action']}")
        print(f"      reason: {s['reason']}")


def write_suggestion_file(suggestions, cfg, season):
    payload = {
        "_advisory": ("SUGGESTION ONLY -- NOT config.json. A human reviews this "
                      "and edits config.json manually if warranted. No bet is "
                      "placed. 1-800-GAMBLER."),
        "breakeven_-110": round(BREAKEVEN, 6),
        "min_n": MIN_N,
        "margin": MARGIN,
        "season_filter": season,
        "current_thresholds": cfg["thresholds"],
        "suggestions": suggestions,
    }
    os.makedirs(os.path.dirname(SUGGESTION_PATH), exist_ok=True)
    with open(SUGGESTION_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\n   wrote advisory suggestion file -> {SUGGESTION_PATH}")


# -------------------------------------------------------------------------- #
# main                                                                       #
# -------------------------------------------------------------------------- #
def run(conn, season=None, write=False) -> int:
    cfg = _load_config()
    df = _graded_picks(conn, season)

    if df.empty:
        print("no graded picks yet -- the selector thresholds are starting "
              "defaults; tune once live weeks accrue.")
        print("(offseason with an empty warehouse: this is the expected path.)")
        print("\ncurrent shipped/config thresholds (probability-edge bars):")
        for market, bars in cfg["thresholds"].items():
            print(f"   {market:<18} lean={bars['lean']:.3f}  "
                  f"playable={bars['playable']:.3f}  strong={bars['strong']:.3f}  "
                  f"ev_min={bars['ev_min']:.3f}")
        print("\nadvisory only; leans, not locks; no bet placement. 1-800-GAMBLER.")
        return 0

    scope = f"season={season}" if season is not None else "all seasons"
    print(f"graded ACTIVE picks: n={len(df)}  ({scope})")
    print("NOTE: picks-hit is the SYNTHETIC-LINE signal (model beating its own")
    print("line); CLV below is the real edge signal. Advisory only. 1-800-GAMBLER.")

    print_by_tier(df)
    print_by_market_tier(df)
    print_edge_bins(df, cfg)
    print_clv_by_tier(conn, df, season)

    suggestions = build_suggestions(df, cfg)
    print_suggestions(suggestions)
    if write:
        write_suggestion_file(suggestions, cfg, season)
    else:
        print("\n   (re-run with --write to emit "
              "data/selector_tuning_suggestion.json)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int, default=None,
                    help="restrict to a single season (default: all)")
    ap.add_argument("--write", action="store_true",
                    help="also write data/selector_tuning_suggestion.json (advisory)")
    ap.add_argument("--db", default=None,
                    help="override DB path (defaults to the warehouse)")
    args = ap.parse_args()

    conn = dbmod.connect(args.db) if args.db else dbmod.connect()
    try:
        return run(conn, season=args.season, write=args.write)
    finally:
        conn.close()


if __name__ == "__main__":
    # keep DEFAULT_SELECTOR_CFG referenced so the config source is explicit
    assert "thresholds" in DEFAULT_SELECTOR_CFG
    raise SystemExit(main())
