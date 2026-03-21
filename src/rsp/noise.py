import numpy as np
import numpy.random as nr
from scipy import fft
from typing import Union
from . import constants as c


def unity_variance_complex_noise(inSize: Union[tuple, int]):
    """
    Generates complex Gaussian noise with unity variance.

    Args:
        inSize (Union[tuple, int]): Shape of the output array.

    Returns:
        np.ndarray: A complex array where the real and imaginary components are
            independent standard normal distributions, scaled to achieve unit variance.
    """
    return (nr.standard_normal(size=inSize) + 1j * nr.standard_normal(size=inSize)) / np.sqrt(2)


def band_limited_complex_noise(f_min, f_max, N_samples, fs, normalize=False):
    """
    Generates band-limited complex noise within a specified frequency range.

    Args:
        f_min (float): Minimum frequency limit in Hz.
        f_max (float): Maximum frequency limit in Hz.
        N_samples (int): Total number of time-domain samples to generate.
        fs (float): Sampling frequency in Hz.
        normalize (bool, optional): If True, normalizes the output signal to unit
            magnitude. Defaults to False.

    Returns:
        np.ndarray: A complex time-domain array representing the band-limited noise.

    Raises:
        AssertionError: If N_samples is not an integer, f_min > f_max, or fs <= 0.
    """
    assert isinstance(N_samples, int), "Samples must be an integer."
    assert f_min <= f_max
    assert fs > 0

    freqs = fft.fftfreq(N_samples, 1 / fs)
    f = np.zeros(N_samples, dtype=np.complex64)
    indices = np.where(np.logical_and(freqs >= f_min, freqs <= f_max))[0]
    random_phase = 2 * np.pi * np.random.rand(indices.size)

    f[indices] = np.exp(1j * random_phase)
    noise = fft.ifft(f)

    if normalize:
        return noise / np.abs(noise)
    else:
        return noise


def guassian_complex_noise(mu, sigma, p, N_samples, fs, normalize=False):
    """
    Generates complex noise with a Power Spectral Density (PSD) shaped by a Gaussian envelope.

    Args:
        mu (float): Center frequency of the Gaussian PSD in Hz.
        sigma (float): Standard deviation of the Gaussian distribution.
        p (float): Order of the Gaussian distribution (p=1 for standard Gaussian).
        N_samples (int): Total number of samples to generate.
        fs (float): Sampling frequency in Hz.
        normalize (bool, optional): If True, normalizes the output signal to unit
            magnitude. Defaults to False.

    Returns:
        np.ndarray: A complex time-domain array with Gaussian-shaped spectral content.

    Raises:
        AssertionError: If N_samples is not an integer or fs <= 0.
    """
    assert isinstance(N_samples, int), "Samples must be an integer."
    assert fs > 0

    freqs = fft.fftfreq(N_samples, 1 / fs)
    f = np.zeros(N_samples, np.complex64)
    f = 1 / (sigma * np.sqrt(2 * c.PI)) * np.exp(-(((freqs - mu) ** 2 / (2 * sigma**2)) ** p)) + 0j

    for i in range(f.size):
        f[i] *= np.exp(1j * 2 * c.PI * np.random.rand())

    noise = fft.ifft(f) * np.sqrt(N_samples)

    if normalize:
        return noise / np.abs(noise)
    else:
        return noise
