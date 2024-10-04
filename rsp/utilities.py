import numpy as np
from . import constants as c


def phase_negpi_pospi(phase: list):
    """return input phase in [-pi, pi)"""

    if not hasattr(phase, "__iter__"):
        phase = [phase]

    phase = np.array(phase)
    phase = phase % (2 * c.PI)
    # phase = phase.tolist()
    for i, p in enumerate(phase):
        if p >= c.PI:
            phase[i] = p - 2 * c.PI
    return phase


def phase_zero_twopi(phase: list):
    """return input phase in [0, 2pi)"""
    phase = np.array(phase)
    phase = phase % (2 * c.PI)
    return phase
