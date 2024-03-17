import numpy as np
from scipy import fft

def band_limited_complex_noise(min_freq, max_freq, samples, sampleRate, normalize=False):
    freqs = fft.fftfreq(samples, 1/sampleRate)
    f = np.zeros(samples, np.complex64)
    indices = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    random_phase = 2*np.pi*np.random.rand(indices.size)

    #noise with random phase (needed)

    f[indices] = np.exp(1j*random_phase)

    noise = fft.ifft(f)

    if normalize:
        return noise/abs(noise)
    else:
        return noise
