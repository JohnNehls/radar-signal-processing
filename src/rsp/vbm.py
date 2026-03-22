import numpy as np
from numpy.linalg import norm
import logging
from . import constants as c
from .noise import band_limited_complex_noise, gaussian_complex_noise
from .waveform import lfm_pulse

logger = logging.getLogger(__name__)


####################################################################################################
### Start: noise techniques to achieve VBM in order of complexity ###
####################################################################################################
def _random_phase(N_pulses, *args):
    """Generates a sequence of complex numbers with random phase.

    This function creates a noise sequence where each element has a unit
    magnitude and a phase uniformly distributed between 0 and 2*pi. This
    results in broadband noise across the entire frequency spectrum.

    Args:
        N_pulses: The number of pulses, determining the length of the noise sequence.
        *args: Catches unused arguments to maintain a consistent interface with
               other noise functions.

    Returns:
        A 1D numpy array of complex numbers with shape (N_pulses,), representing
        the random phase noise.
    """
    rand_phase = 2 * c.PI * np.random.rand(N_pulses)
    return np.exp(1j * rand_phase)


def _uniform_bandwidth_phase(N_pulses, f_delta, PRF):
    """Generates band-limited noise with a uniform spectral distribution.

    This creates a complex noise sequence whose power is concentrated within a
    specified frequency band [-f_delta/2, +f_delta/2]. The noise has a uniform
    distribution in the frequency domain within this band.

    Args:
        N_pulses: The number of pulses, determining the length of the noise sequence.
        f_delta: The total bandwidth of the noise in Hertz.
        PRF: The Pulse Repetition Frequency in Hertz.

    Returns:
        A 1D numpy array of complex numbers with shape (N_pulses,), representing
        the band-limited noise. The output is normalized to have unit power.
    """
    return band_limited_complex_noise(-f_delta / 2, +f_delta / 2, N_pulses, PRF, normalize=True)


def _gaussian_bandwidth_phase(N_pulses, f_delta, PRF):
    """Generates band-limited noise with a Gaussian spectral distribution.

    This creates a complex noise sequence whose power is concentrated within a
    specified frequency band, following a Gaussian (normal) distribution centered
    at 0 Hz with a standard deviation of f_delta / 2.

    Args:
        N_pulses: The number of pulses, determining the length of the noise sequence.
        f_delta: The total bandwidth of the noise in Hertz. This is used to define
                 the standard deviation of the Gaussian spectrum.
        PRF: The Pulse Repetition Frequency in Hertz.

    Returns:
        A 1D numpy array of complex numbers with shape (N_pulses,), representing
        the Gaussian band-limited noise. The output is normalized to have unit power.
    """
    return gaussian_complex_noise(0, f_delta / 2, 1, N_pulses, PRF, normalize=True)


def _gaussian_bandwidth_phase_normalized(N_pulses, f_delta, PRF):
    """Generates Gaussian noise, then normalizes the sequence magnitude.

    This function first generates a complex noise sequence with a Gaussian
    spectral distribution. It then normalizes the entire time-domain sequence
    so that its L2 norm is equal to the square root of the number of pulses.
    This ensures the total energy is conserved.

    Args:
        N_pulses: The number of pulses, determining the length of the noise sequence.
        f_delta: The total bandwidth of the noise in Hertz.
        PRF: The Pulse Repetition Frequency in Hertz.

    Returns:
        A 1D numpy array of complex numbers with shape (N_pulses,) representing
        the normalized Gaussian noise.
    """
    slowtime_noise = gaussian_complex_noise(0, f_delta / 2, 1, N_pulses, PRF, normalize=False)
    slowtime_noise = slowtime_noise / norm(slowtime_noise) * np.sqrt(slowtime_noise.size)
    return slowtime_noise


def _lfm_phase(N_pulses, f_delta, PRF):
    """Generates a slow-time LFM waveform to be used as VBM noise.

    This method creates a Linear Frequency Modulated (LFM) pulse across the
    slow-time dimension. The frequency sweeps across the bandwidth f_delta over
    the total coherent processing interval. This is considered a "clean" method
    as it distributes energy uniformly across the desired Doppler band.

    Args:
        N_pulses: The number of pulses, determining the length of the sequence.
        f_delta: The frequency sweep bandwidth in Hertz.
        PRF: The Pulse Repetition Frequency in Hertz.

    Returns:
        A 1D numpy array of complex numbers with shape (N_pulses,) representing
        the LFM phase sequence.
    """
    _, slowtime_noise = lfm_pulse(PRF, f_delta, N_pulses / PRF, 1, normalize=False)
    return slowtime_noise


####################################################################################################
### End: noise techniques to achieve VBM in order of complexity ###
####################################################################################################


def calc_f_delta(fcar, rdot_delta):
    """Calculates the Doppler frequency spread from a range-rate spread.

    This function converts a spread in target radial velocities (range-rate delta)
    into a corresponding Doppler frequency bandwidth.

    Args:
        fcar: The carrier frequency of the radar in Hertz.
        rdot_delta: The spread in range-rates (radial velocities) in meters/second.

    Returns:
        The corresponding frequency spread (f_delta) in Hertz.
    """
    return 2 * fcar / c.C * rdot_delta


def slowtime_noise(N_pulses, fcar, rdot_delta, PRF, noiseFun=_lfm_phase):
    """Generates a slow-time noise sequence for Velocity Band Modulation (VBM).

    This function acts as a wrapper to generate various types of complex noise
    sequences applied across pulses (slow-time). It first calculates the required
    Doppler frequency spread from the given range-rate spread and then calls the
    specified noise generation function.

    Args:
        N_pulses: The total number of pulses in the coherent processing interval.
        fcar: The radar's carrier frequency in Hertz.
        rdot_delta: The desired spread in range-rates (radial velocities) in m/s,
            which defines the VBM bandwidth.
        PRF: The Pulse Repetition Frequency in Hertz.
        noiseFun: The function to use for generating the noise sequence.
            Defaults to `_lfm_phase`.

    Returns:
        A 1D numpy array representing the complex slow-time noise sequence.

    Notes:
        Available `noiseFun` choices include:
        - `_random_phase`: Broadband random phase noise.
        - `_uniform_bandwidth_phase`: Noise with a uniform spectral distribution.
        - `_gaussian_bandwidth_phase`: Noise with a Gaussian spectral distribution.
        - `_gaussian_bandwidth_phase_normalized`: Gaussian noise with time-domain normalization.
        - `_lfm_phase`: A deterministic LFM sweep across the Doppler band (cleanest).
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
