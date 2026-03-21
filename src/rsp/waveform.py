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
    """Generates a simple, baseband rectangular (uncoded) pulse.

    Args:
        sampleRate (float): The sampling rate in Hz.
        BW (float): The pulse bandwidth in Hz. The pulse duration is 1/BW.
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The real-valued magnitude of the pulse samples.

    Raises:
        AssertionError: If the sample rate is below the Nyquist rate (2 * BW).
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
    """Generates a complex-valued pulse with a constant frequency offset.

    This function creates a rectangular pulse and modulates it with a complex
    exponential at a given carrier frequency.

    Args:
        sampleRate (float): The sampling rate in Hz.
        BW (float): The pulse bandwidth in Hz. The pulse duration is 1/BW.
        fc (float): The carrier frequency offset in Hz.
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag_c (np.ndarray): The complex-valued samples of the pulse.

    Raises:
        AssertionError: If the sample rate is below the Nyquist rate (2 * BW).
    """
    t, mag = uncoded_pulse(sampleRate, BW, normalize=normalize)
    mag_c = np.exp(2j * c.PI * fc * t) * mag
    return t, mag_c


def coded_pulse(sampleRate, BW, code, normalize=True):
    """Generates a baseband, phase-coded pulse.

    The pulse is constructed by concatenating rectangular "chips", where each
    chip has a phase determined by the input code (1 for 0 phase, -1 for pi
    phase).

    Args:
        sampleRate (float): The sampling rate in Hz.
        BW (float): The chip bandwidth in Hz. The chip duration is 1/BW.
        code (list[int]): A list of code values, which must be either 1 or -1.
            The length of the list determines the number of chips.
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The real-valued, coded magnitude of the pulse.

    Raises:
        AssertionError: If any value in the code is not 1 or -1.
    """
    nChips = len(code)
    Tc = 1 / BW
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
    """Generates a baseband, Barker-coded pulse.

    Barker codes are specific binary phase codes known for their low
    autocorrelation sidelobes.

    Args:
        sampleRate (float): The sampling rate in Hz.
        BW (float): The chip bandwidth in Hz. The chip duration is 1/BW.
        nChips (int): The number of chips in the Barker code. Must be a valid
            Barker code length (2, 3, 4, 5, 7, 11, or 13).
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The real-valued, Barker-coded magnitude of the pulse.

    Raises:
        AssertionError: If nChips is not a valid Barker code length.
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
    """Generates a baseband pulse with a random binary phase code.

    The code consists of a sequence of randomly chosen 1s and -1s.

    Args:
        sampleRate (float): The sampling rate in Hz.
        BW (float): The chip bandwidth in Hz. The chip duration is 1/BW.
        nChips (int): The number of chips in the random code.
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The real-valued, randomly coded magnitude.
    """
    code_rand = np.random.choice([1, -1], size=nChips)
    return coded_pulse(
        sampleRate,
        BW,
        code_rand,
        normalize=normalize,
    )


def lfm_pulse(sampleRate, BW, T, chirpUpDown, normalize=True):
    """Generates a baseband Linear Frequency Modulated (LFM) pulse (chirp).

    The instantaneous frequency of the pulse varies linearly with time over the
    pulse duration.

    Args:
        sampleRate (float): The sampling rate in Hz.
        BW (float): The bandwidth of the frequency sweep in Hz.
        T (float): The total pulse duration in seconds.
        chirpUpDown (int): Determines the direction of the frequency sweep.
            1 for an up-chirp (frequency increases), -1 for a down-chirp
            (frequency decreases).
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The complex-valued samples of the LFM pulse.

    Raises:
        AssertionError: If chirpUpDown is not 1 or -1.
    """
    assert chirpUpDown in [1, -1], "ValueError: chirpUpDown must be either 1 or -1."
    dt = 1 / sampleRate
    t = np.arange(0, T, dt)
    k = BW / T  # Chirp rate
    phase = chirpUpDown * np.pi * k * (t**2) - chirpUpDown * np.pi * BW * t
    mag = np.exp(1j * phase)

    if normalize:
        mag = mag / norm(mag)

    return t, mag
