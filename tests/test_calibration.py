import numpy as np
import pytest

from nflvalue.calibration import binary_calibration


def test_perfect_calibration_has_zero_penalty():
    result = binary_calibration([0, 0, 1, 1], [0, 0, 1, 1], bins=2)
    assert result["n"] == 4
    assert result["ece"] == 0
    assert result["overconfidence_ece"] == 0
    assert result["brier"] == 0


def test_overconfidence_is_penalized_and_weighted():
    result = binary_calibration([0, 0, 1, 1], [0.9, 0.9, 0.9, 0.9], bins=10)
    assert result["ece"] == pytest.approx(0.4)
    assert result["overconfidence_ece"] == pytest.approx(0.4)
    assert sum(row["n"] for row in result["table"]) == 4


def test_missing_pairs_are_explicitly_excluded():
    result = binary_calibration([0, 1, np.nan], [0.2, 0.8, 0.4])
    assert result["n"] == 2


@pytest.mark.parametrize(
    "outcomes,probabilities",
    [([0, 2], [0.2, 0.8]), ([0, 1], [-0.1, 0.8]), ([0], [0.2, 0.3])],
)
def test_invalid_calibration_inputs_fail_closed(outcomes, probabilities):
    with pytest.raises(ValueError):
        binary_calibration(outcomes, probabilities)
