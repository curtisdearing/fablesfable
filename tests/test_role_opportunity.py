"""Tests for nflvalue.role_opportunity (current-season vs historical-prior pooling).

Every frame here is a STRICT SYNTHETIC FIXTURE built in this file -- no
pinned or production data is read, and nothing touches the network.
"""

from __future__ import annotations

import copy
import datetime as dt
import inspect
import math

import pandas as pd
import pytest

from nflvalue import role_opportunity as ro

UTC = dt.timezone.utc
AS_OF = dt.datetime(2025, 9, 20, 12, 0, tzinfo=UTC)


def _game(season, week, pid, team, pos, targets=0, carries=0, pass_attempts=0,
          team_pass=35, team_rush=25, rec_yards=None, receptions=None,
          rush_yards=None, pass_yards=None, game_id=None):
    return dict(
        season=season, week=week, game_id=game_id or f"{season}_{week:02d}_{team}",
        player_id=pid, team=team, position=pos,
        targets=targets, receptions=receptions if receptions is not None else round(targets * 0.6),
        rec_yards=rec_yards if rec_yards is not None else targets * 8.0,
        carries=carries, rush_yards=rush_yards if rush_yards is not None else carries * 4.0,
        pass_attempts=pass_attempts,
        pass_yards=pass_yards if pass_yards is not None else pass_attempts * 7.0,
        team_pass_att=team_pass, team_rush_att=team_rush,
    )


def _fixture():
    """SYNTHETIC: WR 'A' (same team both seasons), WR 'B' (changes team),
    WR 'R' (no prior season), RB 'C'. 2024 = prior season, 2025 W1-2 current."""
    rows = []
    for w in range(1, 18):
        rows.append(_game(2024, w, "A", "AAA", "WR", targets=7))          # share 0.2
        rows.append(_game(2024, w, "B", "BBB", "WR", targets=3))
        rows.append(_game(2024, w, "C", "AAA", "RB", targets=2, carries=15))
    for w in (1, 2):
        rows.append(_game(2025, w, "A", "AAA", "WR", targets=14))         # share 0.4
        rows.append(_game(2025, w, "B", "CCC", "WR", targets=10))
        rows.append(_game(2025, w, "R", "AAA", "WR", targets=3))
        rows.append(_game(2025, w, "C", "AAA", "RB", targets=1, carries=20))
    return pd.DataFrame(rows)


def _targets():
    return pd.DataFrame([
        dict(player_id=p, team=t, position=pos, game_id=f"2025_03_{t}")
        for p, t, pos in [("A", "AAA", "WR"), ("B", "CCC", "WR"), ("R", "AAA", "WR"), ("C", "AAA", "RB")]
    ])


def _hyper():
    """SYNTHETIC hand-set hyperparameters so the closed form can be checked."""
    cell = lambda k, rho, mu, mu_new, k0: dict(k=k, rho=rho, phi=k * rho, mu=mu, mu_new=mu_new, k0=k0,
                                             n_player_seasons=100, pooled=False)
    h = {"version": ro.MODEL_VERSION, "fit_seasons": [2023, 2024], "quantities": {}}
    for q, pos_list in ro.QUANTITY_POSITIONS.items():
        h["quantities"][q] = {}
        for pos in pos_list:
            h["quantities"][q][pos] = {
                "same_team": cell(100.0, 0.01, 0.10, 0.05, 50.0),
                "team_changed": cell(20.0, 0.05, 0.10, 0.05, 50.0),
                "no_prior": cell(10.0, 0.10, 0.10, 0.05, 50.0),
            }
    for q in ro.TEAM_QUANTITIES:
        h["quantities"][q] = {"TEAM": {"team": cell(4.0, 10.0, 34.0, 34.0, 8.0)}}
    return h


def _run(games=None, targets=None, hyper=None, **kw):
    return ro.forecast_opportunity(
        games if games is not None else _fixture(),
        targets if targets is not None else _targets(),
        season=2025, week=3, as_of=kw.pop("as_of", AS_OF),
        hyper=hyper or _hyper(), **kw)


def _player(out, pid):
    return next(p for p in out["players"] if p["player_id"] == pid)


def test_as_of_must_be_timezone_aware():
    with pytest.raises(ValueError, match="timezone"):
        _run(as_of=dt.datetime(2025, 9, 20, 12, 0))


def test_post_kickoff_as_of_is_rejected():
    t = _targets()
    t["game_start"] = pd.Timestamp("2025-09-20T11:00:00Z")
    with pytest.raises(ValueError, match="kickoff"):
        _run(targets=t)


def test_target_week_and_later_rows_cannot_change_the_forecast():
    base = _run()
    g = _fixture()
    leak = pd.DataFrame([_game(2025, 3, "A", "AAA", "WR", targets=30),
                         _game(2025, 4, "A", "AAA", "WR", targets=30)])
    after = _run(games=pd.concat([g, leak], ignore_index=True))
    assert _player(after, "A")["target_share"] == _player(base, "A")["target_share"]


def test_rows_with_game_start_after_as_of_are_excluded_and_counted():
    g = _fixture()
    g["game_start"] = pd.Timestamp("2025-09-10T17:00:00Z")
    g.loc[(g["season"] == 2025) & (g["week"] == 2), "game_start"] = pd.Timestamp("2025-09-21T17:00:00Z")
    out = _run(games=g)
    assert out["meta"]["rows_excluded_by_clock"] == int(((g["season"] == 2025) & (g["week"] == 2)).sum())
    assert _player(out, "A")["target_share"]["current_games"] == 1


def test_posterior_is_the_closed_form_precision_blend():
    out = _run()
    ts = _player(out, "A")["target_share"]
    h = _hyper()["quantities"]["target_share"]["WR"]["same_team"]
    prior = (17 * 35 * 0.2 + h["k0"] * h["mu"]) / (17 * 35 + h["k0"])
    d_cur = 70.0
    assert ts["regime"] == "same_team"
    assert ts["prior_source"] == "historical_prior_prev_season"
    assert ts["prior_estimate"] == pytest.approx(prior)
    assert ts["current_estimate"] == pytest.approx(28 / 70)
    assert ts["current_denominator"] == d_cur
    assert ts["k"] == h["k"]
    assert ts["w_current"] == pytest.approx(d_cur / (d_cur + h["k"]))
    assert ts["posterior"] == pytest.approx((28 + h["k"] * prior) / (d_cur + h["k"]))
    g = prior * (1 - prior)
    assert ts["posterior_sd"] == pytest.approx(math.sqrt(h["rho"] * g * h["k"] / (h["k"] + d_cur)))


def test_weight_on_current_grows_with_support_and_stays_below_one():
    g = _fixture()
    ws = []
    for last_week in (1, 2):
        sub = g[~((g["season"] == 2025) & (g["week"] > last_week))]
        ws.append(_player(_run(games=sub), "A")["target_share"]["w_current"])
    assert 0 < ws[0] < ws[1] < 1


def test_team_changer_uses_its_own_regime_strength_not_a_boost():
    ts = _player(_run(), "B")["target_share"]
    assert ts["regime"] == "team_changed"
    assert ts["k"] == _hyper()["quantities"]["target_share"]["WR"]["team_changed"]["k"]
    lo, hi = sorted([ts["prior_estimate"], ts["current_estimate"]])
    assert lo <= ts["posterior"] <= hi   # a convex blend: never beyond the evidence


def test_no_prior_season_is_labelled_position_mean_not_a_preseason_projection():
    ts = _player(_run(), "R")["target_share"]
    assert ts["regime"] == "no_prior"
    assert ts["prior_source"] == "position_mean_no_prior"
    assert ts["prior_estimate"] == 0.05
    assert ts["prior_games"] == 0


def test_zero_current_games_is_missing_not_zero():
    g = _fixture()
    g = g[g["season"] == 2024]
    ts = _player(_run(games=g), "A")["target_share"]
    assert ts["current_games"] == 0
    assert ts["current_estimate"] is None
    assert ts["w_current"] == 0.0
    assert ts["posterior"] == pytest.approx(ts["prior_estimate"])


def test_split_rows_within_one_game_count_as_one_game():
    g = _fixture()
    a = g[(g["player_id"] == "A") & (g["season"] == 2025) & (g["week"] == 1)].iloc[0].to_dict()
    half = dict(a, targets=7, receptions=4, rec_yards=56.0)
    g = pd.concat([g[~((g["player_id"] == "A") & (g["season"] == 2025) & (g["week"] == 1))],
                   pd.DataFrame([half, half])], ignore_index=True)
    ts = _player(_run(games=g), "A")["target_share"]
    assert ts["current_games"] == 2
    assert ts["current_denominator"] == 70.0   # team attempts counted once per game


def test_team_share_conservation_scales_only_when_over_one():
    h = _hyper()
    for reg in h["quantities"]["target_share"]["WR"].values():
        reg["k"] = 1e-3   # effectively current-only
    g = _fixture()
    extra = pd.DataFrame([_game(2025, w, "D", "AAA", "WR", targets=30) for w in (1, 2)])
    t = pd.concat([_targets(), pd.DataFrame([dict(player_id="D", team="AAA", position="WR",
                                                  game_id="2025_03_AAA")])], ignore_index=True)
    out = _run(games=pd.concat([g, extra], ignore_index=True), targets=t, hyper=h)
    team = out["teams"]["AAA"]
    shares = [_player(out, p)["target_share"]["posterior_conserved"] for p in ("A", "R", "C", "D")]
    assert team["target_share_raw_sum"] > 1
    assert sum(shares) == pytest.approx(1.0)
    assert team["target_share_scale"] < 1
    base = _run()
    assert base["teams"]["AAA"]["target_share_scale"] == 1.0
    assert base["teams"]["AAA"]["target_share_unallocated"] > 0


def test_expected_volume_is_team_volume_times_conserved_share():
    out = _run()
    p = _player(out, "A")
    team = out["teams"]["AAA"]
    assert p["expected_targets"] == pytest.approx(
        team["team_pass_att"]["posterior"] * p["target_share"]["posterior_conserved"])
    assert p["expected_targets_prior_only"] == pytest.approx(
        team["team_pass_att"]["prior_estimate"] * p["target_share"]["prior_estimate"])
    assert p["expected_targets_unconserved"] == pytest.approx(
        team["team_pass_att"]["posterior"] * p["target_share"]["posterior"])


def test_routes_and_snaps_are_unavailable_never_proxied_by_targets():
    out = _run()
    recs = [f for f in out["factors"] if f["entity_id"] == "A"]
    by = {f["feature_name"]: f for f in recs}
    for name in ("routes_run", "snaps"):
        assert by[name]["measurement_kind"] == "unavailable"
        assert by[name]["status"] == "unavailable_unverified"
        assert by[name]["observation"] is None
        assert by[name]["consumed"] is False


def test_factor_records_carry_contract_fields_and_default_to_shadow():
    out = _run()
    need = {"factor_id", "category", "entity_id", "entity_type", "game_id", "as_of", "observation",
            "value", "unit", "source_id", "verified", "cutoff_ok", "measurement_kind", "status",
            "model_version", "feature_name", "consumed", "numerical_effect", "numerical_effect_unit",
            "numerical_effect_method", "support_games", "support_opportunities", "reason_not_applied",
            "rationale", "uncertainty"}
    for f in out["factors"]:
        assert need <= set(f)
        assert f["status"] != "numeric_applied"
    ts = next(f for f in out["factors"] if f["entity_id"] == "A" and f["feature_name"] == "expected_targets")
    p = _player(out, "A")
    assert ts["numerical_effect"] == pytest.approx(p["expected_targets"] - p["expected_targets_prior_only"])


def test_primary_consumption_marks_applied_records():
    out = _run(consumption="primary")
    ts = next(f for f in out["factors"] if f["entity_id"] == "A" and f["feature_name"] == "expected_targets")
    assert ts["status"] == "numeric_applied" and ts["consumed"] is True


def test_inputs_are_not_mutated():
    g, t, h = _fixture(), _targets(), _hyper()
    g0, t0, h0 = g.copy(), t.copy(), copy.deepcopy(h)
    _run(games=g, targets=t, hyper=h)
    pd.testing.assert_frame_equal(g, g0)
    pd.testing.assert_frame_equal(t, t0)
    assert h == h0


def test_fit_uses_only_seasons_before_the_cutoff():
    rows = []
    for s in (2022, 2023, 2024):
        for w in range(1, 18):
            for i in range(40):
                rows.append(_game(s, w, f"W{i}", f"T{i % 8}", "WR", targets=(i % 9) + (w % 3) + (s % 2)))
    g = pd.DataFrame(rows)
    h1 = ro.fit_hyperparameters(g, before_season=2024)
    g2 = g.copy()
    g2.loc[g2["season"] == 2024, "targets"] = 0
    h2 = ro.fit_hyperparameters(g2, before_season=2024)
    assert h1 == h2
    assert h1["fit_seasons"] == [2022, 2023]
    cell = h1["quantities"]["target_share"]["WR"]["same_team"]
    assert cell["k"] > 0 and cell["phi"] > 0


def test_no_network_or_io_in_pure_module():
    src = inspect.getsource(ro)
    for bad in ("requests", "urllib", "socket", "read_parquet", "open("):
        assert bad not in src
