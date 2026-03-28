import numpy as np
import pytest
import rsp.uniform_linear_arrays as ula


def test_uniform_weights_center():
    # Uniform weights → phase center at mean of positions
    positions = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert ula.array_phase_center(positions, weights) == pytest.approx(1.0)


def test_single_element():
    # Single element → phase center at that element's position
    positions = np.array([5.0])
    weights = np.array([1.0])
    assert ula.array_phase_center(positions, weights) == pytest.approx(5.0)


def test_asymmetric_weights_shift():
    # Heavier weight on left → phase center shifts left of center
    positions = np.array([0.0, 1.0, 2.0])
    weights = np.array([2.0, 1.0, 0.0])
    # sum(|w|*pos) / sum(w) = (0 + 1 + 0) / 3 = 1/3
    assert ula.array_phase_center(positions, weights) == pytest.approx(1.0 / 3.0)


def test_symmetric_weights_symmetric_positions():
    # Symmetric array with symmetric weights → phase center at geometric center
    positions = np.array([-1.0, 0.0, 1.0])
    weights = np.array([2.0, 1.0, 2.0])
    # sum(2*1 + 1*0 + 2*1) / sum(2+1+2) = 4/5 = 0.8... wait
    # sum(|w|*pos) = 2*1 + 1*0 + 2*1 = 4, sum(w) = 5 → 4/5
    # Hmm, positions are [-1, 0, 1]:
    # sum(2*|-1| ... no, it's abs(w)*pos not abs(w)*abs(pos)
    # sum(abs(w)*pos) = abs(2)*(-1) + abs(1)*(0) + abs(2)*(1) = -2 + 0 + 2 = 0
    # sum(w) = 5
    # result = 0/5 = 0.0 → center
    assert ula.array_phase_center(positions, weights) == pytest.approx(0.0)
