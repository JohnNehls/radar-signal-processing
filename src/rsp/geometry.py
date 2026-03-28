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
    R_vec = np.array(
        [tgt_pos[0] - plat_pos[0], tgt_pos[1] - plat_pos[1], tgt_pos[2] - plat_pos[2]]
    )
    R_unit_vec = R_vec / norm(R_vec)
    R_mag = np.sqrt(R_vec[0] ** 2 + R_vec[1] ** 2 + R_vec[2] ** 2)
    R_dot = np.dot(tgt_vel, R_unit_vec) - np.dot(plat_vel, R_unit_vec)

    return R_vec, R_mag, R_dot
