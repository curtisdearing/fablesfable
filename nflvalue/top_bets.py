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

# Gate tier admission on the Wilson score-interval LOWER BOUND, not the point
# estimate: a thin, lucky band (e.g. 14/20 = 70% but 95% LB ~48%) must NOT clear
# a 67% tier. This is the multi-season-recal lever's "re-derive thresholds with
# proper CIs / never relax" requirement, enforced per band. Wider n tightens the
# LB toward the point estimate, so accruing seasons can only ADMIT more, never
# relax the bar. z=1.96 (95%).
_WILSON_Z = 1.96


def wilson_lower_bound(wins: int, n: int, z: float = _WILSON_Z) -> Optional[float]:
    """Lower bound of the Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None
    phat = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - margin) / denom)


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
        band: {
            "accuracy": (sum(v) / len(v)) if v else None,
            "accuracy_lb": wilson_lower_bound(sum(v), len(v)),
            "wins": sum(v), "n": len(v),
        }
        for band, v in sorted(grades.items())
    }


def _band_for(score: float, kind: str, bands: Dict[str, Dict]) -> Optional[Dict]:
    table = EDGE_BANDS if kind == "edge" else PROB_BANDS
    for lo, hi in table:
        if _in_band(score, lo, hi):
            key = _band_label(lo, hi, kind)
            rec = bands.get(key)
            if rec and rec["n"] >= 20 and rec["accuracy"] is not None:
                return {"band": key, "accuracy": rec["accuracy"],
                        "accuracy_lb": rec.get("accuracy_lb"), "n": rec["n"]}
            return {"band": key, "accuracy": None, "accuracy_lb": None,
                    "n": (rec or {}).get("n", 0)}
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
        b = _band_for(c["score"], c["kind"], bands) or {"band": None, "accuracy": None, "accuracy_lb": None, "n": 0}
        ranked.append({**c, "band": b["band"], "band_accuracy": b["accuracy"],
                       "band_accuracy_lb": b.get("accuracy_lb"), "band_n": b["n"]})
    bets, rank = [], 0
    for c in ranked:
        # Admit on the CI LOWER BOUND so thin/lucky bands cannot qualify.
        acc_lb = c["band_accuracy_lb"]
        tier = None
        if acc_lb is not None and rank < BEST_MAX_RANK and acc_lb >= BEST_ACC:
            tier = "best"
        elif acc_lb is not None and rank < ALL_MAX_RANK and acc_lb > VALUE_ACC and c["edge"] > 0:
            tier = "value"
        if tier:
            rank += 1
            bets.append({"rank": rank, "tier": tier, **{k: c[k] for k in
                        ("market", "selection", "band", "band_accuracy",
                         "band_accuracy_lb", "band_n", "edge")}})
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
            "best_rule": f"rank<=5 and band accuracy 95% LB >={BEST_ACC:.0%}",
            "value_rule": f"rank<=10 and band accuracy 95% LB >{VALUE_ACC:.0%} and edge>0",
            "fail_closed": "games show fewer bets when bands do not qualify; thresholds never relax",
            "calibration": "settled graded picks in data/weekly.json; per-band Wilson 95% LB gates admission (thin/lucky bands excluded). Populating multi-season graded replays into weekly.json remains data-gated.",
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
