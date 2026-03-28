import math
from dataclasses import dataclass, field
from . import constants as c


@dataclass
class Radar:
    """Pulse-Doppler radar system parameters.

    All gain/loss fields are linear (not dB).
    Npulses is computed automatically from dwell_time and PRF.
    """

    fcar: float        # Carrier frequency [Hz]
    txPower: float     # Transmit power [W]
    txGain: float      # Transmit antenna gain [linear]
    rxGain: float      # Receive antenna gain [linear]
    opTemp: float      # Operating temperature [K]
    sampRate: float    # Sampling rate [Hz]
    noiseFactor: float # Receiver noise factor [linear]
    totalLosses: float # Total system losses [linear]
    PRF: float         # Pulse repetition frequency [Hz]
    dwell_time: float  # Coherent dwell time [s]
    Npulses: int = field(init=False)  # Computed from dwell_time and PRF

    def __post_init__(self):
        self.Npulses = int(math.ceil(self.dwell_time * self.PRF))


def range_unambiguous(PRF):
    """
    Calculate the maximum unambiguous range for a given pulse repetition frequency.

    Args:
        PRF (float): Pulse repetition frequency [Hz]

    Returns:
        float: Maximum unambiguous range [m]
    """
    return c.C / (2 * PRF)


def range_resolution(B):
    """
    Calculate the slant range resolution based on the signal bandwidth.

    Args:
        B (float): Pulse bandwidth [Hz]

    Returns:
        float: Range resolution [m]
    """
    return c.C / (2 * B)


def range_aliased(range, PRF):
    """
    Calculate the apparent target range as it appears after range wrap-around.

    Args:
        range (float): True target range [m]
        PRF (float): Pulse repetition frequency [Hz]

    Returns:
        float: Aliased (apparent) range [m]
    """
    return range % range_unambiguous(PRF)


def frequency_delta_doppler(rangeRate, f0):
    """
    Calculate the Doppler frequency shift resulting from target motion.

    Args:
        rangeRate (float): Target range rate (radial velocity) [m/s]
        f0 (float): Carrier frequency of the radar pulse [Hz]

    Returns:
        float: Doppler frequency shift [Hz]
    """
    return f0 * (-2 * rangeRate / c.C)


def frequency_aliased(freq, fs):
    """
    Calculate the apparent frequency after aliasing into the Nyquist interval [-fs/2, fs/2].

    Args:
        freq (float): Input frequency [Hz]
        fs (float): Sampling frequency [Hz]

    Returns:
        float: Aliased frequency within [-fs/2, fs/2] [Hz]
    """
    f = freq % fs
    if f > fs / 2:
        return f - fs
    else:
        return f


def rangeRate_pm_unambiguous(PRF, f0):
    """
    Calculate the unambiguous radial velocity (range rate) limits (+/-).

    Args:
        PRF (float): Pulse repetition frequency [Hz]
        f0 (float): Carrier frequency of the radar pulse [Hz]

    Returns:
        float: Maximum unambiguous range rate magnitude [m/s]
    """
    return PRF * c.C / (4 * f0)


def rangeRate_aliased_rrmax(rangeRate, rangeRate_max):
    """
    Calculate the apparent range rate after aliasing within specified velocity bounds.

    Args:
        rangeRate (float): True range rate [m/s]
        rangeRate_max (float): Maximum unambiguous range rate magnitude [m/s]

    Returns:
        float: Aliased range rate within [-rangeRate_max, rangeRate_max] [m/s]
    """
    r = rangeRate % (2 * rangeRate_max)
    if r > rangeRate_max:
        return r - 2 * rangeRate_max
    else:
        return r


def rangeRate_aliased_prf_f0(rangeRate, PRF, f0):
    """
    Calculate the apparent range rate after aliasing based on system parameters.

    Args:
        rangeRate (float): True range rate [m/s]
        PRF (float): Pulse repetition frequency [Hz]
        f0 (float): Carrier frequency of the radar pulse [Hz]

    Returns:
        float: Aliased range rate [m/s]
    """
    return rangeRate_aliased_rrmax(rangeRate, rangeRate_pm_unambiguous(PRF, f0))


def first_echo_pulse_bin(range, PRF):
    """
    Determine the pulse index (slow-time bin) in which the first target return arrives.

    Args:
        range (float): True target range [m]
        PRF (float): Pulse repetition frequency [Hz]

    Returns:
        int: The pulse interval index (0-indexed) [unitless]
    """
    return int(range / range_unambiguous(PRF))
