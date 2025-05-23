import numpy as np
from . import constants as c


def power2db(power: float):
    """
    Convert power in Watts to dB
    Args:
        power (float) : Power [W]
    Return:
        power (float) : Power [dB]
    """
    return 10 * np.log10(power)


def db2power(db: float):
    """
    Convert power in dB to Watts
    Args:
        power (float) : Power [dB]
    Return:
        power (float) : Power [W]
    """
    return 10 ** (db / 10)


def volt2db(voltage: float):
    """
    Convert voltage in Volts in Amps to dB
    Args:
        voltage (float) : Voltage in [A]
    Return:
        voltage (float) : Voltage in [dB]
    """
    return 20 * np.log10(voltage)


def db2volt(db: float):
    """
    Convert voltage in dB to Volts
    Args:
        voltage (float) : Voltage in [dB]
    Return:
        voltage (float) : Voltage in [A]
    """
    return 10 ** (db / 20)


def phase_negpi_pospi(phase):
    """
    Place the input phases into [-pi, pi).
    Args:
        phase (float) : Phase
    Return:
        phase (float) : Phase
    """

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
    """
    Place the input phases into [0, 2pi).
    Args:
        phase (float) : Phase
    Return:
        phase (float) : Phase
    """
    phase = np.array(phase)
    phase = phase % (2 * c.PI)
    return phase


def zero_to_smallest_float(array, value=1e-16):
    """
    Set all zero elements of input array to value in place.
    Args:
        array : Input array
        value : Value to set the array's zero values to
    Return:
        None
    """
    indxs = np.where(array == 0)
    array[indxs] = value
