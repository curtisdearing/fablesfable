"""SD provenance, and an honest measure of how much SD error moves a price.

`candidates.market_residual_sd` fits ONE residual SD per market across every
player: on the 2026 Week 1 board D.Maye and S.Darnold both carried
`sd: 97.375` passing yards. That is worth saying out loud on every row --
the SD was never estimated for this player.

WHAT THIS IS NOT. A first pass at this assumed the pooled SD was what made
Darnold's "+4.2% edge" (model 0.5428 vs market 0.5013 on U228.5) untrustworthy.
The arithmetic says otherwise and is worth recording so nobody re-derives the
wrong story: with mean 218.036 and line 228.5, z is only 0.107, and near z=0
the normal CDF is almost flat in SD. Scaling that SD by +/-25% moves P(under)
by ~0.014 -- LESS than the edge. Probability is least sensitive to SD exactly
where the line sits on the mean.

The sensitivity is largest at moderate |z|, which is where a pooled SD really
does distort a price: a tighter SD (sd=25 on the same row) swings ~0.049 under
the same stress. So this module reports the swing per row instead of assuming
which rows are fragile.

None of this changes a probability. Whether SD should be conditioned per
player is a one-lever experiment under docs/ACCURACY_PROTOCOL.md, registered
in analysis/accuracy_protocol.json, and is NOT decided here.
"""

from __future__ import annotations

from nflvalue import prop_decision as pd_


def _qb(sd=97.375, mean=218.036, line=228.5):
    return {"mean": mean, "sd": sd, "line": line, "dist": "normal",
            "market": "passing_yards"}


def test_probability_is_nearly_flat_in_sd_when_the_line_sits_on_the_mean():
    """The corrective result: the real Darnold row is ROBUST to SD error."""
    swing = pd_.sd_stress(_qb())
    assert swing is not None
    assert swing < 0.02, "z=0.107 puts this row on the flat part of the CDF"
    assert pd_.edge_survives_sd_uncertainty(0.042, swing) is True, (
        "the measured +4.2% edge is larger than the swing a 25% SD error "
        "buys -- pooled SD is NOT what makes this row weak")


def test_sensitivity_peaks_away_from_the_mean_not_at_the_widest_sd():
    """A tighter SD on the same row is MORE sd-sensitive, not less, because it
    moves z off zero. Guards against the intuition that widest == most fragile."""
    wide = pd_.sd_stress(_qb(sd=97.375))
    tight = pd_.sd_stress(_qb(sd=25.0))
    assert tight > wide


def test_a_fragile_row_is_flagged():
    """Moderate |z| plus a thin edge: exactly the shape a pooled SD distorts."""
    cand = _qb(sd=25.0)
    swing = pd_.sd_stress(cand)
    assert pd_.edge_survives_sd_uncertainty(0.02, swing) is False


def test_a_genuinely_large_edge_still_survives():
    assert pd_.edge_survives_sd_uncertainty(0.40, pd_.sd_stress(_qb(sd=25.0))) is True


def test_unknown_inputs_are_not_a_pass():
    """Absence of the check must never read as having passed it."""
    assert pd_.edge_survives_sd_uncertainty(None, 0.01) is None
    assert pd_.edge_survives_sd_uncertainty(0.10, None) is None
    assert pd_.sd_stress({"mean": 1.0, "line": 1.5, "dist": "normal"}) is None
    assert pd_.sd_stress(_qb(sd=0.0)) is None


def test_stress_is_the_larger_direction_not_an_average():
    """An upper bound on the swing, so the bad side is never hidden."""
    cand = _qb()
    swing = pd_.sd_stress(cand)
    base = pd_.probability_from_projection(cand)
    both = []
    for scale in (0.75, 1.25):
        probe = dict(cand); probe["sd"] = cand["sd"] * scale
        both.append(abs(pd_.probability_from_projection(probe) - base))
    assert abs(swing - max(both)) < 1e-12
    assert swing >= min(both)


def test_composite_reports_sd_provenance_and_the_swing():
    import inspect
    from nflvalue import composite
    src = inspect.getsource(composite)
    for field in ("sd_scope", "sd_prob_swing", "edge_survives_sd_uncertainty",
                  "sd_stress_fraction"):
        assert f'"{field}"' in src, f"{field} must be visible on the row"
