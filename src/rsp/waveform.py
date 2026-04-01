from enum import StrEnum

import numpy as np
from numpy.linalg import norm
from . import constants as c


class WaveformType(StrEnum):
    UNCODED = "uncoded"
    BARKER  = "barker"
    RANDOM  = "random"
    LFM     = "lfm"

BARKER_DICT = {
    2: [1, -1],
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def uncoded_pulse(
    sampleRate: float, BW: float, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
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
    mag = np.ones(t.size)

    if normalize:
        mag = mag / norm(mag)

    return t, mag


def complex_tone_pulse(
    sampleRate: float, BW: float, fc: float, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
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


def coded_pulse(
    sampleRate: float, BW: float, code: list[int], normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
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
    nchips = len(code)
    Tc = 1 / BW
    dt = 1 / sampleRate
    samplesPerChip = round(Tc * sampleRate)

    code_array = np.asarray(code)
    assert np.all(np.abs(code_array) == 1), "ValueError: Code values must be either 1 or -1."
    mag = np.repeat(code_array, samplesPerChip).astype(float)
    t = np.arange(mag.size) * dt

    if normalize:
        mag = mag / norm(mag)

    return t, mag


def barker_coded_pulse(
    sampleRate: float, BW: float, nchips: int, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a baseband, Barker-coded pulse.

    Barker codes are specific binary phase codes known for their low
    autocorrelation sidelobes.

    Args:
        sampleRate (float): The sampling rate in Hz.
        BW (float): The chip bandwidth in Hz. The chip duration is 1/BW.
        nchips (int): The number of chips in the Barker code. Must be a valid
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
    assert nchips in BARKER_DICT, f"Error: {nchips=} is not a valid Barker code."
    return coded_pulse(
        sampleRate,
        BW,
        BARKER_DICT[nchips],
        normalize=normalize,
    )


def random_coded_pulse(
    sampleRate: float, BW: float, nchips: int, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a baseband pulse with a random binary phase code.

    The code consists of a sequence of randomly chosen 1s and -1s.

    Args:
        sampleRate (float): The sampling rate in Hz.
        BW (float): The chip bandwidth in Hz. The chip duration is 1/BW.
        nchips (int): The number of chips in the random code.
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The real-valued, randomly coded magnitude.
    """
    code_rand = np.random.choice([1, -1], size=nchips)
    return coded_pulse(
        sampleRate,
        BW,
        code_rand,
        normalize=normalize,
    )


def lfm_pulse(
    sampleRate: float, BW: float, T: float, chirpUpDown: int, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
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


# --- Waveform dict factories ---------------------------------------------------

def uncoded_waveform(bw: float) -> dict[str, object]:
    """Returns a waveform dict for an uncoded rectangular pulse.

    Args:
        bw: Pulse bandwidth in Hz. Pulse duration is 1/bw.

    Returns:
        Waveform dict for use with rdm.gen.
    """
    return {"type": WaveformType.UNCODED, "bw": bw}


def barker_coded_waveform(bw: float, nchips: int) -> dict[str, object]:
    """Returns a waveform dict for a Barker-coded pulse.

    Args:
        bw: Chip bandwidth in Hz. Chip duration is 1/bw.
        nchips: Number of chips. Must be a valid Barker length (2,3,4,5,7,11,13).

    Returns:
        Waveform dict for use with rdm.gen.
    """
    assert nchips in BARKER_DICT, f"nchips={nchips} is not a valid Barker code length."
    return {"type": WaveformType.BARKER, "bw": bw, "nchips": nchips}


def random_coded_waveform(bw: float, nchips: int) -> dict[str, object]:
    """Returns a waveform dict for a random binary phase-coded pulse.

    Args:
        bw: Chip bandwidth in Hz. Chip duration is 1/bw.
        nchips: Number of chips.

    Returns:
        Waveform dict for use with rdm.gen.
    """
    return {"type": WaveformType.RANDOM, "bw": bw, "nchips": nchips}


def lfm_waveform(bw: float, T: float, chirpUpDown: int) -> dict[str, object]:
    """Returns a waveform dict for a Linear Frequency Modulated (LFM) pulse.

    Args:
        bw: Bandwidth of the frequency sweep in Hz.
        T: Pulse duration in seconds.
        chirpUpDown: 1 for up-chirp, -1 for down-chirp.

    Returns:
        Waveform dict for use with rdm.gen.
    """
    assert chirpUpDown in [1, -1], "chirpUpDown must be 1 (up) or -1 (down)."
    return {"type": WaveformType.LFM, "bw": bw, "T": T, "chirpUpDown": chirpUpDown}
