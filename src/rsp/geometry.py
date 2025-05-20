import numpy as np
from numpy.linalg import norm


def range_and_rangerate(plat_pos: list, plat_vel: list, tgt_pos: list, tgt_vel: list):
    """
    Calculate the range vector, range, and range-rate of a target relative to a platform.
    Args:
        plat_pos (list) : Platform position [x,y,z]
        plat_vel (list) : Platform velocity [v_x,v_y,v_z]
        tgt_pos (list) : target position [x,y,z]
        tgt_vel (list) : target velocity [v_x,v_y,v_z]
    Return:
        Range_vector, Range, RangeRate_vector
    """
    R_vec = np.array(
        [tgt_pos[0] - plat_pos[0], tgt_pos[1] - plat_pos[1], tgt_pos[2] - plat_pos[2]]
    )
    R_unit_vec = R_vec / norm(R_vec)
    R_mag = np.sqrt(R_vec[0] ** 2 + R_vec[1] ** 2 + R_vec[2] ** 2)
    R_dot = np.dot(tgt_vel, R_unit_vec) - np.dot(plat_vel, R_unit_vec)

    return R_vec, R_mag, R_dot
