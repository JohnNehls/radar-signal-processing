import math
from dataclasses import dataclass, field
from . import constants as c


@dataclass
class Radar:
    """Pulse-Doppler radar system parameters.

    All gain/loss fields are linear (not dB).
    n_pulses is computed automatically from dwell_time and prf.
    """

    fcar: float        # Carrier frequency [Hz]
    tx_power: float    # Transmit power [W]
    tx_gain: float     # Transmit antenna gain [linear]
    rx_gain: float     # Receive antenna gain [linear]
    op_temp: float     # Operating temperature [K]
    sample_rate: float   # Sampling rate [Hz]
    noise_factor: float # Receiver noise factor [linear]
    total_losses: float # Total system losses [linear]
    prf: float         # Pulse repetition frequency [Hz]
    dwell_time: float  # Coherent dwell time [s]
    n_pulses: int = field(init=False)  # Computed from dwell_time and prf

    def __post_init__(self) -> None:
        """Compute n_pulses from dwell_time and prf."""
        self.n_pulses = int(math.ceil(self.dwell_time * self.prf))


def range_unambiguous(prf: float) -> float:
    """
    Calculate the maximum unambiguous range for a given pulse repetition frequency.

    Args:
        prf (float): Pulse repetition frequency [Hz]

    Returns:
        float: Maximum unambiguous range [m]
    """
    return c.C / (2 * prf)


def range_resolution(B: float) -> float:
    """
    Calculate the slant range resolution based on the signal bandwidth.

    Args:
        B (float): Pulse bandwidth [Hz]

    Returns:
        float: Range resolution [m]
    """
    return c.C / (2 * B)


def range_aliased(range: float, prf: float) -> float:
    """
    Calculate the apparent target range as it appears after range wrap-around.

    Args:
        range (float): True target range [m]
        prf (float): Pulse repetition frequency [Hz]

    Returns:
        float: Aliased (apparent) range [m]
    """
    return range % range_unambiguous(prf)


def frequency_delta_doppler(range_rate: float, f0: float) -> float:
    """
    Calculate the Doppler frequency shift resulting from target motion.

    Args:
        range_rate (float): Target range rate (radial velocity) [m/s]
        f0 (float): Carrier frequency of the radar pulse [Hz]

    Returns:
        float: Doppler frequency shift [Hz]
    """
    return f0 * (-2 * range_rate / c.C)


def frequency_aliased(freq: float, fs: float) -> float:
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
    return f


def range_rate_pm_unambiguous(prf: float, f0: float) -> float:
    """
    Calculate the unambiguous radial velocity (range rate) limits (+/-).

    Args:
        prf (float): Pulse repetition frequency [Hz]
        f0 (float): Carrier frequency of the radar pulse [Hz]

    Returns:
        float: Maximum unambiguous range rate magnitude [m/s]
    """
    return prf * c.C / (4 * f0)


def range_rate_aliased_rrmax(range_rate: float, range_rate_max: float) -> float:
    """
    Calculate the apparent range rate after aliasing within specified velocity bounds.

    Args:
        range_rate (float): True range rate [m/s]
        range_rate_max (float): Maximum unambiguous range rate magnitude [m/s]

    Returns:
        float: Aliased range rate within [-range_rate_max, range_rate_max] [m/s]
    """
    r = range_rate % (2 * range_rate_max)
    if r > range_rate_max:
        return r - 2 * range_rate_max
    return r


def range_rate_aliased_prf_f0(range_rate: float, prf: float, f0: float) -> float:
    """
    Calculate the apparent range rate after aliasing based on system parameters.

    Args:
        range_rate (float): True range rate [m/s]
        prf (float): Pulse repetition frequency [Hz]
        f0 (float): Carrier frequency of the radar pulse [Hz]

    Returns:
        float: Aliased range rate [m/s]
    """
    return range_rate_aliased_rrmax(range_rate, range_rate_pm_unambiguous(prf, f0))


def first_echo_pulse_bin(range: float, prf: float) -> int:
    """
    Determine the pulse index (slow-time bin) in which the first target return arrives.

    Args:
        range (float): True target range [m]
        prf (float): Pulse repetition frequency [Hz]

    Returns:
        int: The pulse interval index (0-indexed) [unitless]
    """
    return int(range / range_unambiguous(prf))
