"""Accuracy-tiered per-game bet ranking.

Per game: "Best Bets" = up to top 5, "More Value" = up to rank 10.
Rank is driven by measured accuracy: a bet enters the BEST tier only when its
confidence band's graded historical accuracy is >= 67%; VALUE requires > 50%
band accuracy and positive edge. Fail-closed: if a game lacks qualifying
candidates the tiers show fewer entries — thresholds are never relaxed to
reach 5/10. Every tier carries its measured band accuracy and n.

Calibration source: settled picks in data/weekly.json (graded replay).
Multi-season walk-forward recalibration is a registered TODO; band provenance
is embedded in the artifact so the dashboard can display it honestly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

BEST_ACC = 0.67
VALUE_ACC = 0.50
BEST_MAX_RANK = 5
ALL_MAX_RANK = 10
# Predeclared edge bands (points of model edge vs line) for ATS/total picks,
# and win-probability bands for straight-up/ML picks.
EDGE_BANDS = [(4.0, None), (2.5, 4.0), (1.5, 2.5), (0.5, 1.5), (0.0, 0.5)]
PROB_BANDS = [(0.70, None), (0.62, 0.70), (0.55, 0.62), (0.50, 0.55)]


def _band_label(lo, hi, unit):
    return f"{unit}>={lo}" if hi is None else f"{unit} {lo}-{hi}"


def _in_band(x, lo, hi):
    return x is not None and x >= lo and (hi is None or x < hi)


def calibrate_bands(weekly: Dict) -> Dict[str, Dict]:
    """Measure graded accuracy per predeclared band from settled games."""
    grades: Dict[str, List[int]] = {}
    for wk in weekly.get("weeks", []):
        for g in wk.get("games", []):
            if not g.get("settled"):
                continue
            ats, tot = g.get("ats_pick") or {}, g.get("total_pick") or {}
            if ats.get("edge") is not None and g.get("ats_result") in ("W", "L"):
                for lo, hi in EDGE_BANDS:
                    if _in_band(abs(ats["edge"]), lo, hi):
                        grades.setdefault(_band_label(lo, hi, "edge"), []).append(
                            1 if g["ats_result"] == "W" else 0)
                        break
            if tot.get("edge") is not None and g.get("total_result") in ("W", "L"):
                for lo, hi in EDGE_BANDS:
                    if _in_band(abs(tot["edge"]), lo, hi):
                        grades.setdefault(_band_label(lo, hi, "edge"), []).append(
                            1 if g["total_result"] == "W" else 0)
                        break
            p = g.get("p_home_win")
            if p is not None and g.get("su_correct") is not None:
                p_pick = max(p, 1.0 - p)
                for lo, hi in PROB_BANDS:
                    if _in_band(p_pick, lo, hi):
                        grades.setdefault(_band_label(lo, hi, "p"), []).append(
                            1 if g.get("su_correct") else 0)
                        break
    return {
        band: {"accuracy": (sum(v) / len(v)) if v else None, "n": len(v)}
        for band, v in sorted(grades.items())
    }


def _band_for(score: float, kind: str, bands: Dict[str, Dict]) -> Optional[Dict]:
    table = EDGE_BANDS if kind == "edge" else PROB_BANDS
    for lo, hi in table:
        if _in_band(score, lo, hi):
            key = _band_label(lo, hi, kind)
            rec = bands.get(key)
            if rec and rec["n"] >= 20 and rec["accuracy"] is not None:
                return {"band": key, "accuracy": rec["accuracy"], "n": rec["n"]}
            return {"band": key, "accuracy": None, "n": (rec or {}).get("n", 0)}
    return None


def _candidates(g: Dict) -> List[Dict]:
    out = []
    ats, tot = g.get("ats_pick") or {}, g.get("total_pick") or {}
    if ats.get("side"):
        out.append({"market": "spread", "selection": f"{ats.get('team', ats['side'])} {ats.get('line')}",
                    "score": abs(ats.get("edge") or 0.0), "kind": "edge",
                    "edge": ats.get("edge") or 0.0})
    if tot.get("side"):
        out.append({"market": "total", "selection": f"{tot['side']} {tot.get('line')}",
                    "score": abs(tot.get("edge") or 0.0), "kind": "edge",
                    "edge": tot.get("edge") or 0.0})
    p = g.get("p_home_win")
    if p is not None and g.get("su_pick"):
        out.append({"market": "moneyline", "selection": g["su_pick"],
                    "score": max(p, 1.0 - p), "kind": "p",
                    "edge": max(p, 1.0 - p) - 0.5})
    return out


def rank_game(g: Dict, bands: Dict[str, Dict]) -> List[Dict]:
    ranked = []
    for c in sorted(_candidates(g), key=lambda c: (-(c["score"] if c["kind"] == "edge" else c["score"] * 10), c["market"])):
        b = _band_for(c["score"], c["kind"], bands) or {"band": None, "accuracy": None, "n": 0}
        ranked.append({**c, "band": b["band"], "band_accuracy": b["accuracy"], "band_n": b["n"]})
    bets, rank = [], 0
    for c in ranked:
        acc = c["band_accuracy"]
        tier = None
        if acc is not None and rank < BEST_MAX_RANK and acc >= BEST_ACC:
            tier = "best"
        elif acc is not None and rank < ALL_MAX_RANK and acc > VALUE_ACC and c["edge"] > 0:
            tier = "value"
        if tier:
            rank += 1
            bets.append({"rank": rank, "tier": tier, **{k: c[k] for k in
                        ("market", "selection", "band", "band_accuracy", "band_n", "edge")}})
    return bets


def build_top_bets(weekly: Dict) -> Dict:
    bands = calibrate_bands(weekly)
    weeks = []
    for wk in weekly.get("weeks", []):
        games = []
        for g in wk.get("games", []):
            bets = rank_game(g, bands)
            if bets:
                games.append({"home": g.get("home"), "away": g.get("away"),
                              "settled": bool(g.get("settled")), "bets": bets})
        weeks.append({"week": wk.get("week"), "label": wk.get("label"), "games": games})
    return {
        "meta": {
            "best_rule": f"rank<=5 and band accuracy>={BEST_ACC:.0%}",
            "value_rule": f"rank<=10 and band accuracy>{VALUE_ACC:.0%} and edge>0",
            "fail_closed": "games show fewer bets when bands do not qualify; thresholds never relax",
            "calibration": "settled graded picks in data/weekly.json; multi-season walk-forward recalibration pending",
            "bands": bands,
        },
        "weeks": weeks,
    }


def main(root: Optional[Path] = None) -> Path:
    root = root or Path(__file__).resolve().parent.parent
    weekly = json.loads((root / "data" / "weekly.json").read_text())
    out = build_top_bets(weekly)
    dest = root / "data" / "top_bets.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    return dest


if __name__ == "__main__":
    print(main())
