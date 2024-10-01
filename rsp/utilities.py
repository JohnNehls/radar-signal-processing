import numpy as np
from .constants import PI, C, K_BOLTZ

def phase_negpi_pospi(phase: list):
    """return phase in [-pi, pi)"""

    if not hasattr(phase, '__iter__'):
        phase = [phase]

    phase = np.array(phase)
    phase = phase%(2*PI)
    # phase = phase.tolist()
    for i, p in enumerate(phase):
        if p >= PI:
            phase[i] = p - 2*PI
    return phase

def phase_zero_twopi(phase: list):
    """return phase in [-pi, pi)"""
    phase = np.array(phase)
    phase = phase%(2*PI)
    return phase
