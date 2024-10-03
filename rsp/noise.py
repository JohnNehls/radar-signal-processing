import numpy as np
from scipy import fft
from . import constants as c

def band_limited_complex_noise(min_freq, max_freq, samples, sampleRate, normalize=False):
    freqs = fft.fftfreq(samples, 1/sampleRate)
    f = np.zeros(samples, np.complex64)
    indices = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    random_phase = 2*np.pi*np.random.rand(indices.size)

    #noise with random phase (needed)
    f[indices] = np.exp(1j*random_phase)

    noise = fft.ifft(f)

    if normalize:
        # return noise/norm(noise)*np.sqrt(samples)
        # TODO! maybe just multiply each by df?
        return noise/abs(noise)
    else:
        return noise

def guassian_complex_noise(mu, sigma, p, samples, sampleRate, normalize=False):
    freqs = fft.fftfreq(samples, 1/sampleRate)
    f = np.zeros(samples, np.complex64)
    f = 1/(sigma*np.sqrt(2*c.PI))*np.exp(-((freqs-mu)**2/(2*sigma**2))**p) + 0j

    for i in range(f.size):
        f[i] *= np.exp(1j*2*c.PI*np.random.rand())

    noise = fft.ifft(f)*np.sqrt(samples)

    if normalize:
        # return noise/norm(noise)*np.sqrt(samples)
        # TODO! maybe just multiply each by df?
        return noise/abs(noise)
    else:
        return noise
