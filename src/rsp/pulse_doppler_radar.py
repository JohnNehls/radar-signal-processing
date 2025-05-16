from . import constants as c


def range_unambiguous(PRF):
    """
    Unambigious range.
    Args:
        PRF (float) : Pulse repitition frequency [Hz]
    Return:
        range : float [m]
    """
    return c.C / (2 * PRF)


def range_resolution(B):
    """
    Range resolution.
    Args:
        B (float) : Pulse bandwidth [Hz]
    Return:
        range : float [m]
    """
    return c.C / (2 * B)


def range_aliased(range, PRF):
    """
    Target range as it appears after aliasing.
    Args:
        range (float) : Taget range
        PRF (float) : Pulse repitition frequency [Hz]
    Return:
        range : float [m]
    """
    return range % range_unambiguous(PRF)


def frequency_delta_doppler(rangeRate, f0):
    """
    Frequnce delta due to Doppler shift after bouncing off of target.
    Args:
        rangeRate (float) : Target rangeRate
        f0 (float) : Frequency of the radar pulse [Hz]
    Return:
        f_doppler : float [Hz]
    """
    return f0 * (-2 * rangeRate / c.C)


def frequency_aliased(freq, fs):
    """
    Frequency as it appears after being aliasing into [-fs/2, fs/2]
    Args:
        freq (float) : Frequency [Hz]
        fs (float) : Sample frequency [Hz]
    Return:
        freq : float [Hz]
    """
    f = freq % fs
    if f > fs / 2:
        return f - fs
    else:
        return f


def rangeRate_pm_unambiguous(PRF, f0):
    """+/- bounds of the unambgeous velocity.
    Args:
        PRF (float) : Pulse repitition frequency [Hz]
        f0 (float) : Frequency of the radar pulse [Hz]
    Return:
        rangeRate : float [m/s]
    """
    return PRF * c.C / (4 * f0)


def rangeRate_aliased_rrmax(rangeRate, rangeRate_max):
    """
    RangeRate as it appears after aliasing in [-rangeRate_max, rangeRate_max].
    Args:
        rangeRate (float) : Input rangeRate [m/s]
        rangeRate_max (float) : Maximum rangeRate [m/s]
    Return:
        rangeRate : float [m/s]
    """
    r = rangeRate % (2 * rangeRate_max)
    if r > rangeRate_max:
        return r - 2 * rangeRate_max
    else:
        return r


def rangeRate_aliased_prf_f0(rangeRate, PRF, f0):
    """
    RangeRate as it appears after aliasing in [-rangeRate_max, rangeRate_max].
    Args:
        rangeRate (float) : Input rangeRate [m/s]
        PRF (float) : Pulse repitition frequency [Hz]
        f0 (float) : Frequency of the radar pulse [Hz]
    Return:
        rangeRate : float [m/s]
    """
    return rangeRate_aliased_rrmax(rangeRate, rangeRate_pm_unambiguous(PRF, f0))


def first_echo_pulse_bin(range, PRF):
    """
    The slow-time bin the first target return will arrive in.
    Args:
        range (float) : Taget range
        PRF (float) : Pulse repitition frequency [Hz]
    Return:
        bin : int [unitless]
    """
    return int(range / range_unambiguous(PRF))
