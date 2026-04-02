import math
import pytest
import rsp.utilities as utl

PI = math.pi


# --- phase_negpi_pospi wraps to [-pi, pi) ---


@pytest.mark.parametrize(
    "input_phase, expected",
    [
        (0, 0.0),
        (PI / 2, PI / 2),
        (-PI / 2, -PI / 2),
        (PI, -PI),  # pi maps to -pi (boundary: pi is excluded, -pi is included)
        (-PI, -PI),  # -pi stays -pi
        (2 * PI, 0.0),  # full rotation back to 0
        (3 * PI, -PI),  # 3pi → pi → -pi
    ],
)
def test_phase_negpi_pospi(input_phase, expected):
    result = utl.phase_negpi_pospi(input_phase)
    assert result[0] == pytest.approx(expected, abs=1e-10)


# --- phase_zero_twopi wraps to [0, 2pi) ---


@pytest.mark.parametrize(
    "input_phase, expected",
    [
        (0, 0.0),
        (PI, PI),
        (-PI, PI),  # -pi maps to pi
        (2 * PI, 0.0),  # 2pi maps to 0 (boundary: 2pi excluded, 0 included)
        (3 * PI, PI),  # 3pi → pi
        (-PI / 2, 3 * PI / 2),
    ],
)
def test_phase_zero_twopi(input_phase, expected):
    result = utl.phase_zero_twopi(input_phase)
    assert result == pytest.approx(expected, abs=1e-10)
