"""Geometric range, range-rate, and flight-path calculations.

Computes the instantaneous range and range-rate between a radar platform and
a target given their 3-D position and velocity vectors.  Also provides helpers
for SAR flight-path generation and slant-range computation.
"""

import numpy as np
from numpy.linalg import norm


def range_and_rangerate(
    plat_pos: list, plat_vel: list, tgt_pos: list, tgt_vel: list
) -> tuple[np.ndarray, float, float]:
    """
    Calculate the range vector, range magnitude, and range-rate of a target relative to a platform.

    Args:
        plat_pos (list): Cartesian position of the platform [x, y, z].
        plat_vel (list): Velocity vector of the platform [vx, vy, vz].
        tgt_pos (list): Cartesian position of the target [x, y, z].
        tgt_vel (list): Velocity vector of the target [vx, vy, vz].

    Returns:
        R_vec (np.ndarray): The displacement vector from the platform to the target.
        R_mag (float): The Euclidean distance (range) between the platform and target.
        R_dot (float): The scalar range-rate (radial velocity) of the target relative to the platform.
    """
    R_vec = np.array(tgt_pos) - np.array(plat_pos)
    R_mag = norm(R_vec)
    R_unit_vec = R_vec / R_mag
    R_dot = np.dot(tgt_vel, R_unit_vec) - np.dot(plat_vel, R_unit_vec)

    return R_vec, R_mag, R_dot


def flight_path(n_pulses: int, pulse_spacing: float, altitude: float = 0.0) -> np.ndarray:
    """Generates a straight, level flight path centred at the origin.

    The platform moves along the *x*-axis (along-track) at a fixed altitude.
    Positions are centred so that pulse ``n_pulses // 2`` is at *x* = 0.

    Args:
        n_pulses: Number of aperture positions.
        pulse_spacing: Along-track distance between successive positions [m].
        altitude: Platform altitude above the scene plane [m].

    Returns:
        Platform positions with shape ``(n_pulses, 3)``, columns ``[x, y, z]``.
    """
    x = (np.arange(n_pulses) - n_pulses / 2) * pulse_spacing
    positions = np.zeros((n_pulses, 3))
    positions[:, 0] = x
    positions[:, 2] = altitude
    return positions


def slant_range(platform_positions: np.ndarray, target_position: list[float]) -> np.ndarray:
    """Computes the Euclidean slant range from each platform position to a target.

    Args:
        platform_positions: Platform positions with shape ``(n_pulses, 3)`` [m].
        target_position: Target ``[x, y, z]`` coordinates [m].

    Returns:
        1-D array of slant ranges with shape ``(n_pulses,)`` [m].
    """
    diff = platform_positions - np.asarray(target_position)
    return norm(diff, axis=1)
