import numpy as np
from numpy.linalg import norm
from . import constants as c

BARKER_DICT = {
    2: [1, -1],
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def uncoded_pulse(sampleRate, BW, normalize=True):
    """
    Create a baseband uncoded pulse.
    Args:
        sampleRate (float) : Sample rate [Hz]
        BW (float) : Pulse Bandwidth [Hz]
        normalize : bool (optional: default is True)
    Return:
        t, mag : np.array, np.array
    """
    assert sampleRate / BW >= 2, "Error: sample rate below Nyquist"

    T = 1 / BW
    dt = 1 / sampleRate
    t = np.arange(0, T, dt)
    mag = np.zeros(t.size)

    mag[np.where(t <= T)] = 1

    if normalize:
        mag = mag / norm(mag)

    return t, mag


def complex_tone_pulse(sampleRate, BW, fc, normalize=True):
    """
    Create a complex tone pulse.
    Args:
        sampleRate (float) : Sample rate [Hz]
        BW (float) : Pulse Bandwidth [Hz]
        normalize : bool (optional: default is True)
    Return:
        t, mag_c : np.array, np.array
    """
    t, mag = uncoded_pulse(sampleRate, BW, normalize=normalize)
    mag_c = np.exp(2j * c.PI * fc * t) * mag
    return t, mag_c


def coded_pulse(sampleRate, BW, code, normalize=True):
    """
    Create a baseband coded pulse.
    Args:
        sampleRate (float) : Sample rate [Hz]
        BW (float) : Pulse Bandwidth [Hz]
        code (list) : List of code values, values are either 1 or -1
        normalize : bool (optional: default is True)
    Return:
        t, mag : np.array, np.array
    """

    nChips = len(code)

    Tc = 1 / BW
    # T = nChips * Tc
    dt = 1 / sampleRate

    samplesPerChip = round(Tc * sampleRate)

    mag = np.zeros((nChips, samplesPerChip))

    for i, val in enumerate(code):
        assert val == 1 or val == -1, "ValueError: Code value must be either 1 or -1."
        mag[i, :] = val

    mag = mag.flatten()

    t = np.arange(mag.size) * dt

    if normalize:
        mag = mag / norm(mag)

    return t, mag


def barker_coded_pulse(sampleRate, BW, nChips, normalize=True):
    """
    Create a baseband Barker-coded pulse.
    Args:
        sampleRate (float) : Sample rate [Hz]
        BW (float) : Pulse Bandwidth [Hz]
        nChips (int) : The number of chips in the Barker code
        normalize : bool (optional: default is True)
    Return:
        t, mag : np.array, np.array
    """
    assert nChips in BARKER_DICT, f"Error: {nChips=} is not a valid Barker code."
    assert nChips == len(BARKER_DICT[nChips]), "Error: Barker dict is incorrect"
    return coded_pulse(
        sampleRate,
        BW,
        BARKER_DICT[nChips],
        normalize=normalize,
    )


def random_coded_pulse(sampleRate, BW, nChips, normalize=True):
    """
    Create a baseband random-coded pulse.
    Args:
        sampleRate (float) : Sample rate [Hz]
        BW (float) : Pulse Bandwidth [Hz]
        nChips (int) : The number of chips in the Barker code
        normalize : bool (optional: default is True)
    Return:
        t, mag : np.array, np.array
    """
    code_rand = np.random.choice([1, -1], size=nChips)
    return coded_pulse(
        sampleRate,
        BW,
        code_rand,
        normalize=normalize,
    )


def lfm_pulse(sampleRate, BW, T, chirpUpDown, normalize=True):
    """
    Create a baseband LFM pulse.
    Args:
        sampleRate (float) : Sample rate [Hz]
        BW (float) : Pulse Bandwidth [Hz]
        T (int) : Pulse length [s]
        chirpUpDown (int) : Indicates Chirp direction, either 1 or -1.
        normalize : bool (optional: default is True)
    Return:p
        t, mag : np.array, np.array
    """
    assert chirpUpDown == 1 or chirpUpDown == -1, "ValueError: chirpUpDown must be either 1 or -1."
    dt = 1 / sampleRate
    t = np.arange(0, T, dt)
    f = chirpUpDown * (-BW * t + BW * t**2 / T) / 2
    mag = np.zeros(t.size, dtype=np.complex64)
    i_ar = np.where(t <= T)
    mag[i_ar] = np.exp(1j * 2 * c.PI * f[i_ar])

    if normalize:
        mag = mag / norm(mag)

    return t, mag
