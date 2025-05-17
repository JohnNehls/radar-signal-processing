import numpy as np
import numpy.random as nr
from scipy import fft
from typing import Union
from . import constants as c


def unity_variance_complex_noise(inSize: Union[tuple, int]):
    """
    Create complex noise with unity variance
    Args:
        inSize (tuple): Shape of the output array.
    Return:
        noise : Complex array
    """
    return (nr.standard_normal(size=inSize) + 1j * nr.standard_normal(size=inSize)) / np.sqrt(2)


def band_limited_complex_noise(f_min, f_max, N_samples, fs, normalize=False):
    """
    Create band-limited complex noise.
    Args:
        f_min (float) : Minimum frequency [Hz]
        f_max (float) : Maximum frequency [Hz]
        N_samples (int) : number of samples
        fs (float) : Sample rate [Hz]
        normalize (bool) : Normalize out put flag (optional, default is False)
    Return:
        noise : Complex array
    """
    assert type(N_samples) is int, "Samples must be an integer."
    assert f_min <= f_max
    assert fs > 0

    freqs = fft.fftfreq(N_samples, 1 / fs)
    f = np.zeros(N_samples, np.complex64)
    indices = np.where(np.logical_and(freqs >= f_min, freqs <= f_max))[0]
    random_phase = 2 * np.pi * np.random.rand(indices.size)

    # noise with random phase (needed)
    f[indices] = np.exp(1j * random_phase)

    noise = fft.ifft(f)

    if normalize:
        # return noise/norm(noise)*np.sqrt(samples)
        # TODO! maybe just multiply each by df?
        return noise / abs(noise)
    else:
        return noise


def guassian_complex_noise(mu, sigma, p, N_samples, fs, normalize=False):
    """
    Create complex noise with Gaussian PSD.
    Args:
        mu (float) : Center of the Gaussian
        sigma (float) : Standard deviation
        p (float) : order of the Gaussian
        N_samples (int) : number of samples
        fs (float) : Sample rate [Hz]
        normalize (bool) : Normalize out put flag (optional, default is False)
    Return:
        noise : Complex array
    """
    assert type(N_samples) is int, "Samples must be an integer."
    assert fs > 0

    freqs = fft.fftfreq(N_samples, 1 / fs)
    f = np.zeros(N_samples, np.complex64)
    f = 1 / (sigma * np.sqrt(2 * c.PI)) * np.exp(-(((freqs - mu) ** 2 / (2 * sigma**2)) ** p)) + 0j

    for i in range(f.size):
        f[i] *= np.exp(1j * 2 * c.PI * np.random.rand())

    noise = fft.ifft(f) * np.sqrt(N_samples)

    if normalize:
        # return noise/norm(noise)*np.sqrt(samples)
        # TODO! maybe just multiply each by df?
        return noise / abs(noise)
    else:
        return noise
