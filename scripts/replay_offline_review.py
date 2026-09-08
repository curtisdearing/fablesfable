#!/usr/bin/env python3
"""No-external-write, no-network end-to-end replay of the live pipeline.

    python3 scripts/replay_offline_review.py [--out evidence.json]

Runs the Wednesday clock and a T-90 clock on the synthetic two-team week the
test-suite uses, with every feed injected, every write target redirected to a
temp directory, and ``urllib.request.urlopen`` disabled so any accidental
network call fails loudly. Emits an input -> output evidence record: the
candidate pool, every roster exclusion with its reason, the published leans
with their market state / probability source, what the DB persisted, and the
probability-grade denominators. Used by reports/claude_accuracy_review.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _no_network(*a, **k):
    raise AssertionError("network call attempted during the offline replay")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    urllib.request.urlopen = _no_network  # type: ignore[assignment]

    import pipeline_weekly as pw
    from nflvalue import config as cfgmod, db as dbmod, report as rptmod, document as docmod
    from nflvalue.freshness import stamp_now
    from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs
    from tests.test_pipeline_weekly import _fresh_feeds, _roster
    from analysis.prop_probability_grade import grade_rows, load_rows

    tmp = Path(tempfile.mkdtemp(prefix="ff-replay-"))
    db_path = str(tmp / "replay.db")
    real_connect = dbmod.connect
    dbmod.connect = lambda p=None: real_connect(db_path)
    rptmod.REPORTS_DIR = str(tmp / "reports")
    rptmod.WEEKLY_PROPS_JSON = str(tmp / "weekly_props.json")
    docmod.DROPS_DIR = str(tmp / "drops")
    cfgmod.LATEST_PATH = str(tmp / "latest.json")
    cfgmod.DASHBOARD_PATH = str(tmp / "dashboard.html")
    game_id = f"{SEASON}_09_AAA_BBB"
    now = stamp_now()
    inputs = synthetic_inputs()

    # -- INPUT: the carry-forward candidate pool before any live gate
    from nflvalue import candidates as candmod
    pool = candmod.enumerate_candidates(SEASON, WEEK, inputs=inputs, roster_mode="carry_forward")
    ev = {"as_of": now, "season": SEASON, "week": WEEK, "game_id": game_id,
          "input_candidates": pool[["player_id", "name", "team", "market", "mean", "sd",
                                    "line", "dist", "p_over", "line_source"]].to_dict("records")}

    # roster: RB_A traded to ZZZ (carry-forward row still says AAA); QB_B active;
    # WR_A active; a retired ghost row for a player not in the pool
    roster = _roster(now, extra=[{"player_id": "GHOST", "team": "AAA", "status": "RET", "week": WEEK}])
    for r in roster["rows"]:
        if r["player_id"] == "RB_A":
            r["team"] = "ZZZ"
    feeds = _fresh_feeds(now)
    feeds["active_roster"] = roster
    wed = pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=feeds,
                      discord=True, discord_dry_run=True)
    ev["wed"] = {
        "publish": wed["publish"], "publish_reasons": wed["publish_reasons"],
        "roster_gate": wed["roster_gate"], "roster_eligibility": wed["roster_eligibility"],
        "discord": wed["discord"],
        "leans": [{k: ln.get(k) for k in ("player_id", "team", "market", "side", "line",
                                            "line_source", "market_state", "p_over", "p_under",
                                            "ml_score", "ml_p_over", "composite", "edge")}
                  | {"model_prob": (ln.get("components") or {}).get("model_prob"),
                     "model_prob_source": (ln.get("components") or {}).get("model_prob_source"),
                     "ev_best_price": (ln.get("components") or {}).get("ev_best_price"),
                     "kelly_fraction": (ln.get("components") or {}).get("kelly_fraction")}
                  for g in wed["games"] for ln in g["leans"]],
    }

    # -- T-90: WR_A is inactive on the event roster -> voided + re-rank
    feeds_t90 = _fresh_feeds(now)
    feeds_t90["active_roster"] = roster
    feeds_t90["inactive_rows"] = [{"name": "Alpha Wideout", "active": False, "team": "AAA"},
                                  {"name": "Bravo Quarterback", "active": True, "team": "BBB"}]
    feeds_t90["inactives_fetched_at"] = now
    t90 = pw.run_t90(SEASON, WEEK, game_id, mode="live", inputs=inputs, inject_feeds=feeds_t90,
                     discord=True, discord_dry_run=True)
    ev["t90"] = {"publish": t90["publish"], "publish_reasons": t90["publish_reasons"],
                 "voided": t90["voided"], "roster_gate": t90["roster_gate"],
                 "roster_eligibility": t90["roster_eligibility"],
                 "leans": [{k: ln.get(k) for k in ("player_id", "market", "side", "market_state")}
                           for g in t90["games"] for ln in g["leans"]]}

    conn = real_connect(db_path)
    ev["db"] = {
        "user_version": conn.execute("PRAGMA user_version").fetchone()[0],
        "leans": dbmod.query_df(conn, "SELECT clock, game_id, player_id, market, side, line_source, "
                                      "market_state, n_books, p_side, status, void_reason FROM leans "
                                      "ORDER BY clock, player_id, market").to_dict("records"),
        "lines_rows": int(conn.execute("SELECT COUNT(*) FROM lines").fetchone()[0]),
        "api_credits_rows": int(conn.execute("SELECT COUNT(*) FROM api_credits").fetchone()[0]),
    }
    conn.close()
    ev["grade"] = grade_rows(load_rows(db_path), bootstrap_n=50)
    ev["artifacts"] = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file())
    ev["tmp_dir"] = str(tmp)
    out = json.dumps(ev, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
