import sys
import numpy as np
from . import constants as c


def power2db(power):
    return 10 * np.log10(power)


def db2power(db):
    return 10 ** (db / 10)


def volt2db(voltage):
    return 20 * np.log10(voltage)


def db2volt(db):
    return 10 ** (db / 20)


def phase_negpi_pospi(phase):
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


def phase_zero_twopi(phase):
    """return input phase in [0, 2pi)"""
    phase = np.array(phase)
    phase = phase % (2 * c.PI)
    return phase


def zero_to_smallest_float(array):
    """set all elements of input array to smallest float32"""
    indxs = np.where(array == 0)
    array[indxs] = sys.float_info.min
