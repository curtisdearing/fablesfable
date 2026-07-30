"""QB haircut lab: detection walk-forward safety, fit recovery, book consistency."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.qb_haircut import backup_flags, fit_h, BOOK_PATH, MIN_P_IMPROVE


def _sched(rows):
    return pd.DataFrame(rows, columns=[
        "season", "week", "game_type", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name", "gameday"])


def _mk_rows(team_a_qb_by_week, n_weeks):
    """One team ('AAA') hosting a rotating opponent with a stable QB."""
    rows = []
    for w in range(1, n_weeks + 1):
        rows.append((2023, w, "REG", "AAA", f"OP{w}",
                     team_a_qb_by_week[w - 1], f"oppqb{w}", "x", "y",
                     f"2023-09-{w:02d}"))
    return rows


def test_no_flag_before_history_exists():
    """First 8 games can never be flagged (window unfilled -> fail closed)."""
    qbs = ["q1"] * 4 + ["q2"] * 4                 # change at game 5, but window < 8
    flags = backup_flags(_sched(_mk_rows(qbs, 8)), "modal8")
    assert not any(v[0] for v in flags.values())


def test_modal8_flags_change_after_established_incumbent():
    qbs = ["q1"] * 8 + ["q2"]                     # 8 straight, then a new starter
    flags = backup_flags(_sched(_mk_rows(qbs, 9)), "modal8")
    assert flags[(2023, 9, "AAA", "OP9")][0] is True


def test_abrupt_flags_only_first_games_of_absence():
    qbs = ["q1"] * 8 + ["q2", "q2", "q2"]         # abrupt change, then backup settles
    flags = backup_flags(_sched(_mk_rows(qbs, 11)), "abrupt")
    assert flags[(2023, 9, "AAA", "OP9")][0] is True     # first backup start
    # by game 11 the incumbent q1 did NOT start the previous game -> not flagged
    assert flags[(2023, 11, "AAA", "OP11")][0] is False


def test_abrupt_ignores_settled_regime_change():
    """A long-settled new starter is not an 'abrupt absence'."""
    qbs = ["q1"] * 6 + ["q2"] * 6
    flags = backup_flags(_sched(_mk_rows(qbs, 12)), "abrupt")
    assert flags[(2023, 12, "AAA", "OP12")][0] is False


def test_detection_is_strictly_prior():
    """Changing FUTURE starts must not change an earlier game's flag."""
    qbs_a = ["q1"] * 8 + ["q2", "q1"]
    qbs_b = ["q1"] * 8 + ["q2", "q3"]             # differs only at game 10
    fa = backup_flags(_sched(_mk_rows(qbs_a, 10)), "modal8")
    fb = backup_flags(_sched(_mk_rows(qbs_b, 10)), "modal8")
    assert fa[(2023, 9, "AAA", "OP9")] == fb[(2023, 9, "AAA", "OP9")]


def test_fit_h_recovers_planted_haircut():
    rng = np.random.default_rng(5)
    rows = []
    true_h = 3.0
    for i in range(4000):
        bh, ba = (i % 10 == 0), (i % 17 == 0)
        m = float(rng.normal(0, 7))
        y = m - true_h * bh + true_h * ba + float(rng.normal(0, 2))
        rows.append((2022, i % 18 + 1, m, y, bh, ba))
    h, _ = fit_h(rows)
    assert abs(h - true_h) <= 0.5


def test_fit_h_zero_when_no_effect():
    rng = np.random.default_rng(6)
    rows = []
    for i in range(4000):
        m = float(rng.normal(0, 7))
        rows.append((2022, i % 18 + 1, m, m + float(rng.normal(0, 2)),
                     (i % 10 == 0), (i % 17 == 0)))
    h, _ = fit_h(rows)
    assert h <= 0.5


def test_committed_book_fail_closed_consistency():
    if not os.path.exists(BOOK_PATH):
        pytest.skip("book/qb_haircut.json not built")
    with open(BOOK_PATH) as fh:
        book = json.load(fh)
    for name, v in book["variants"].items():
        if not v["gate"]["passed"]:
            assert v["shipped_haircut_points"] is None, name
        else:
            assert v["pooled"]["p_adj_beats_base"] >= MIN_P_IMPROVE, name
