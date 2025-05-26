import numpy as np
from numpy.linalg import norm
import logging
from . import constants as c
from .noise import band_limited_complex_noise, guassian_complex_noise
from .waveform import lfm_pulse

logger = logging.getLogger(__name__)


####################################################################################################
### Start: noise techniques to achieve VBM in order of complexity ###
####################################################################################################
def _random_phase(N_pulses, *args):
    """
    Random phase, placing energy in all frequencies
    Args:
        N_pulses (int) : Number of pulses
    Return:
        random_noise : Noise with shape (N_pulses)
    """
    rand_phase = 2 * c.PI * np.random.rand(N_pulses)
    return np.exp(1j * rand_phase)


def _uniform_bandwidth_phase(N_pulses, f_delta, PRF):
    """
    Random phase within in a bandwidth
    Args:
        N_pulses (int) : Number of pulses
        f_delta (float) : Frequency delta [Hz]
        PRF (float) : Pulse repitition interval [s]
    Return:
        random_noise : Noise with shape (N_pulses)
    Notes:
       - Does not require assumption on processing interval
       - Dirty result if each element is made magnitude = 1
       - Un-normalized (normalized over interval) only makes sense if possible on hardware
       - Adds much of the engery in the f_delta, but also lots of energy in other freqs
    """
    return band_limited_complex_noise(-f_delta / 2, +f_delta / 2, N_pulses, PRF, normalize=True)


def _gaussian_bandwidth_phase(Npulses, f_delta, PRF):
    """
    Random phase in a bandwidth using a Gaussian distribution
    Args:
        N_pulses (int) : Number of pulses
        f_delta (float) : Frequency delta [Hz]
        PRF (float) : Pulse repitition interval [s]
    Return:
        random_noise : Noise with shape (N_pulses)
    Notes:
        - does not require assumption on processing interval
        - dirty result if each element is made magnitude = 1
        - un-normalized (normalized over interval) only makes sense if possible on hardware
    """
    return guassian_complex_noise(0, f_delta / 2, 1, Npulses, PRF, normalize=True)


def _gaussian_bandwidth_phase_normalized(Npulses, f_delta, PRF):
    """
    Random phase normalized over a period
    Args:
        N_pulses (int) : Number of pulses
        f_delta (float) : Frequency delta [Hz]
        PRF (float) : Pulse repitition interval [s]
    Return:
        random_noise : Noise with shape (N_pulses)
    Notes:
       - A way to make the random noise cleaner is to normalize over a an interval
       - use with un-normalized noise
       - requires knowledge of number of pulses? (maybe)
    """
    slowtime_noise = guassian_complex_noise(0, f_delta / 2, 1, Npulses, PRF, normalize=False)
    slowtime_noise = slowtime_noise / norm(slowtime_noise) * np.sqrt(slowtime_noise.size)
    return slowtime_noise


def _lfm_phase(Npulses, f_delta, PRF):
    """
    Phase created from linear frquency modulation in slowtime.
    Args:
        N_pulses (int) : Number of pulses
        f_delta (float) : Frequency delta [Hz]
        PRF (float) : Pulse repitition interval [s]
    Return:
        random_noise : Noise with shape (N_pulses)
    Notes:
        - cleanest VBM method
    """
    _, slowtime_noise = lfm_pulse(PRF, f_delta, Npulses / PRF, 1, normalize=False)
    return slowtime_noise


####################################################################################################
### End: noise techniques to achieve VBM in order of complexity ###
####################################################################################################


def calc_f_delta(fcar, rdot_delta):
    """
    Convert rdot_delta to a frequency delta
    Args:
        fcar : Carrier frequency [Hz]
        rdot_delta : Range rate delta [m/s]
    Retrun:
        f_delta (float) : Frequency delta [Hz]
    """
    return 2 * fcar / c.C * rdot_delta


def slowtime_noise(N_pulses, fcar, rdot_delta, PRF, noiseFun=_lfm_phase):
    """
    Slowtime noise generation wrapper.
    Args:
        N_pulses (int) : Number of pulses
        fcar : Carrier frequency [Hz]
        rdot_delta : Range rate delta [m/s]
        PRF (float) : Pulse repitition interval [s]    
        noiseFun (func) : function for generationg noise (default =_lfm_phase)
    Return:
        slowtime_noise (array)
    Notes:
        - noiseFun choices: random_VBM, uniform_bandwidth_VMB, gaussian_bandwidth_VBM, gaussian_bandwidth_amp_VBM, lfm_VBM
    """
    f_delta = calc_f_delta(fcar, rdot_delta)
    slowtime_noise = noiseFun(N_pulses, f_delta, PRF)

    logger.debug("\nBand Noise:")
    logger.debug(f"\t{norm(slowtime_noise)=}")
    logger.debug(f"\t{slowtime_noise.size=}")
    logger.debug(f"\t{np.mean(abs(slowtime_noise))=}")
    logger.debug(f"\t{np.var(abs(slowtime_noise))=}")
    logger.debug(f"\t{np.mean(np.angle(slowtime_noise))=}")
    logger.debug(f"\t{np.sqrt(np.var(np.angle(slowtime_noise)))=}")

    return slowtime_noise
