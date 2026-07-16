"""Deterministic binary-probability calibration diagnostics.

Calibration is part of the accuracy objective, not a decorative chart.  The
one-sided ``overconfidence_ece`` penalizes bins whose stated P(over) exceeds
the observed over rate; ordinary ECE also catches under-confidence.  Equal-
width bins are pre-registered in ``analysis/accuracy_protocol.json``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def binary_calibration(
    outcomes: Iterable[float], probabilities: Iterable[float], *, bins: int = 10
) -> dict:
    """Return Brier/ECE diagnostics and an auditable bin table.

    Missing pairs are excluded explicitly. Probabilities outside [0, 1] and
    non-binary outcomes fail closed instead of being silently clipped.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    y = np.asarray(list(outcomes), dtype=float)
    p = np.asarray(list(probabilities), dtype=float)
    if y.shape != p.shape:
        raise ValueError("outcomes and probabilities must have the same shape")
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        return {
            "n": 0, "bins": bins, "brier": None, "ece": None,
            "overconfidence_ece": None, "max_calibration_error": None,
            "table": [],
        }
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("outcomes must be binary")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be between 0 and 1")

    # p==1 belongs to the final bin rather than falling off the right edge.
    index = np.minimum((p * bins).astype(int), bins - 1)
    table = []
    ece = overconfidence = 0.0
    max_error = 0.0
    for number in range(bins):
        mask = index == number
        if not mask.any():
            continue
        predicted = float(p[mask].mean())
        observed = float(y[mask].mean())
        gap = predicted - observed
        weight = float(mask.mean())
        ece += weight * abs(gap)
        overconfidence += weight * max(gap, 0.0)
        max_error = max(max_error, abs(gap))
        table.append({
            "bin": number + 1,
            "lower": number / bins,
            "upper": (number + 1) / bins,
            "n": int(mask.sum()),
            "mean_probability": round(predicted, 6),
            "observed_rate": round(observed, 6),
            "gap": round(gap, 6),
        })
    return {
        "n": int(len(y)),
        "bins": bins,
        "brier": round(float(np.mean((p - y) ** 2)), 6),
        "ece": round(ece, 6),
        "overconfidence_ece": round(overconfidence, 6),
        "max_calibration_error": round(max_error, 6),
        "table": table,
    }
