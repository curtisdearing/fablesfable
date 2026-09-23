"""Regression: ratings_current export must not let a retired alias overwrite
the franchise's live rating.

``build_ratings.ABBR`` is many-to-one -- LV/OAK both map to "Las Vegas
Raiders", as do LAC/SD, LA/LAR/STL, JAX/JAC and WAS/WSH. The export loops the
sorted raw identifiers and writes one key per *full franchise name*, so the
last alias written wins. Sorted ascending that is OAK, not LV: the 2026-09-20
run exported "Las Vegas Raiders" with ``abbr: OAK`` and a near-zero net,
burying LV's actual walk-forward rating.

ALL RATING VALUES IN THIS FILE ARE SYNTHETIC UNIT-TEST FIXTURES. They are
invented numbers chosen to be distinguishable (LV vs OAK differ by many
points); they are not measured football inputs and must never be read as
team strength.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_ratings  # noqa: E402

SEASON = 2025  # label only; no real season data is loaded here


def _fixture(values):
    """(teams, off, deff) from a TEST-ONLY {identifier: (off, def)} mapping."""
    teams = sorted(values)
    off = {t: v[0] for t, v in values.items()}
    deff = {t: v[1] for t, v in values.items()}
    return teams, off, deff


def _export(values, last_seen=None):
    teams, off, deff = _fixture(values)
    return build_ratings.current_ratings(teams, off, deff, SEASON, last_seen=last_seen)


def test_retired_alias_does_not_overwrite_current_rating():
    """The documented defect: OAK sorts after LV and clobbered it."""
    values = {"LV": (5.0, 3.0), "OAK": (-9.0, -7.0), "KC": (2.0, 1.0)}
    out = _export(values, last_seen={"KC": 300, "LV": 299, "OAK": 17})

    raiders = out["Las Vegas Raiders"]
    assert raiders["abbr"] == "LV"
    assert raiders["off"] == 5.0
    assert raiders["def"] == 3.0
    assert raiders["net"] == 8.0


@pytest.mark.parametrize(
    "current,retired,name",
    [
        ("LV", "OAK", "Las Vegas Raiders"),
        ("LAC", "SD", "Los Angeles Chargers"),
        ("LA", "STL", "Los Angeles Rams"),
        ("JAX", "JAC", "Jacksonville Jaguars"),
        ("WAS", "WSH", "Washington Commanders"),
    ],
)
def test_every_alias_group_keeps_the_most_recent_identifier(current, retired, name):
    """Not an LV/OAK special case: every many-to-one group behaves the same."""
    out = _export({current: (4.0, 2.0), retired: (-6.0, -5.0)},
                  last_seen={current: 900, retired: 5})
    assert out[name]["abbr"] == current
    assert out[name]["net"] == 6.0


def test_alias_group_collapses_to_one_entry_and_keeps_every_franchise():
    values = {"LV": (5.0, 3.0), "OAK": (-9.0, -7.0),
              "KC": (2.0, 1.0), "DEN": (-1.0, 0.5)}
    out = _export(values, last_seen={"KC": 300, "DEN": 298, "LV": 299, "OAK": 17})
    assert set(out) == {"Las Vegas Raiders", "Kansas City Chiefs", "Denver Broncos"}


def test_non_alias_teams_are_passed_through_unchanged():
    values = {"LV": (5.0, 3.0), "OAK": (-9.0, -7.0), "KC": (2.0, 1.0)}
    out = _export(values, last_seen={"KC": 300, "LV": 299, "OAK": 17})
    assert out["Kansas City Chiefs"] == {
        "abbr": "KC", "off": 2.0, "def": 1.0, "net": 3.0, "season": SEASON,
    }


def test_missing_current_alias_keeps_the_retired_one():
    """A history that only ever saw OAK still exports the franchise."""
    out = _export({"OAK": (-2.0, 1.0), "KC": (2.0, 1.0)},
                  last_seen={"OAK": 40, "KC": 300})
    assert out["Las Vegas Raiders"]["abbr"] == "OAK"
    assert out["Las Vegas Raiders"]["net"] == -1.0


def test_unmapped_identifier_is_exported_under_its_raw_id():
    out = _export({"ZZZ": (1.5, -0.5), "KC": (2.0, 1.0)},
                  last_seen={"ZZZ": 10, "KC": 300})
    assert out["ZZZ"]["abbr"] == "ZZZ"
    assert out["ZZZ"]["net"] == 1.0
    assert out["Kansas City Chiefs"]["abbr"] == "KC"


def test_export_is_deterministic_regardless_of_input_ordering():
    values = {"LV": (5.0, 3.0), "OAK": (-9.0, -7.0), "KC": (2.0, 1.0),
              "LAC": (1.0, 1.0), "SD": (-3.0, -3.0), "ZZZ": (0.5, 0.5)}
    last_seen = {"KC": 300, "LV": 299, "LAC": 298, "ZZZ": 10, "OAK": 17, "SD": 9}
    first = _export(values, last_seen=last_seen)
    shuffled = dict(reversed(list(values.items())))
    second = build_ratings.current_ratings(
        sorted(shuffled), {t: v[0] for t, v in shuffled.items()},
        {t: v[1] for t, v in shuffled.items()}, SEASON, last_seen=last_seen)
    assert first == second
    assert list(first) == list(second)


def test_without_recency_the_declared_current_alias_wins():
    """Callers that pass no ``last_seen`` still must not regress to OAK."""
    out = _export({"LV": (5.0, 3.0), "OAK": (-9.0, -7.0)})
    assert out["Las Vegas Raiders"]["abbr"] == "LV"


def test_tied_recency_falls_back_to_the_declared_current_alias():
    out = _export({"LV": (5.0, 3.0), "OAK": (-9.0, -7.0)},
                  last_seen={"LV": 42, "OAK": 42})
    assert out["Las Vegas Raiders"]["abbr"] == "LV"
