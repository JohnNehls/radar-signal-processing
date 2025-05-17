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
        sampleRate (float) : Sample rate.
        BW (float) : Pulse Bandwidth
        normalize : bool (optional: default is True)
    Return:
        f_doppler : float [Hz]
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
    """complex tone pulse"""
    t, mag = uncoded_pulse(sampleRate, BW, normalize=normalize)
    mag_c = np.exp(2j * c.PI * fc * t) * mag
    return t, mag_c


def coded_pulse(sampleRate, BW, code, normalize=True):
    """baseband coded pulse"""

    nChips = len(code)

    Tc = 1 / BW
    # T = nChips * Tc
    dt = 1 / sampleRate

    samplesPerChip = round(Tc * sampleRate)

    mag = np.zeros((nChips, samplesPerChip))

    for i, val in enumerate(code):
        mag[i, :] = val

    mag = mag.flatten()

    t = np.arange(mag.size) * dt

    if normalize:
        mag = mag / norm(mag)

    return t, mag


def barker_coded_pulse(sampleRate, BW, nChips, normalize=True):
    """baseband Barker coded pulse"""
    assert nChips in BARKER_DICT, f"Error: {nChips=} is not a valid Barker code."
    assert nChips == len(BARKER_DICT[nChips]), "Error: Barker dict is incorrect"
    return coded_pulse(
        sampleRate,
        BW,
        BARKER_DICT[nChips],
        normalize=normalize,
    )


def random_coded_pulse(sampleRate, BW, nChips, normalize=True):
    """baseband random bi-phase coded pulse"""
    code_rand = np.random.choice([1, -1], size=nChips)
    return coded_pulse(
        sampleRate,
        BW,
        code_rand,
        normalize=normalize,
    )


def lfm_pulse(sampleRate, BW, T, chirpUpDown, normalize=True):
    """baseband LFM pulse"""
    dt = 1 / sampleRate
    t = np.arange(0, T, dt)
    f = chirpUpDown * (-BW * t + BW * t**2 / T) / 2
    mag = np.zeros(t.size, dtype=np.complex64)
    i_ar = np.where(t <= T)
    mag[i_ar] = np.exp(1j * 2 * c.PI * f[i_ar])

    if normalize:
        mag = mag / norm(mag)

    return t, mag


## see /tests/function_tests/process_waveform.py for test of this function
def process_waveform_dict(wvf: dict, radar: dict):
    """Fill in wvf dict with "pulse", "time_BW_product", "pulse_width" """
    if wvf["type"] == "uncoded":
        _, pulse_wvf = uncoded_pulse(radar["sampRate"], wvf["bw"])
        wvf["pulse"] = pulse_wvf
        wvf["time_BW_product"] = 1
        wvf["pulse_width"] = 1 / wvf["bw"]

    elif wvf["type"] == "barker":
        _, pulse_wvf = barker_coded_pulse(radar["sampRate"], wvf["bw"], wvf["nchips"])
        wvf["pulse"] = pulse_wvf
        wvf["time_BW_product"] = wvf["nchips"]
        wvf["pulse_width"] = 1 / wvf["bw"] * wvf["nchips"]

    elif wvf["type"] == "random":
        _, pulse_wvf = random_coded_pulse(radar["sampRate"], wvf["bw"], wvf["nchips"])
        wvf["pulse"] = pulse_wvf
        wvf["time_BW_product"] = wvf["nchips"]
        wvf["pulse_width"] = 1 / wvf["bw"] * wvf["nchips"]

    elif wvf["type"] == "lfm":
        _, pulse_wvf = lfm_pulse(radar["sampRate"], wvf["bw"], wvf["T"], wvf["chirpUpDown"])
        wvf["pulse"] = pulse_wvf
        wvf["time_BW_product"] = wvf["bw"] * wvf["T"]
        wvf["pulse_width"] = wvf["T"]

    else:
        raise Exception(f"wvf type {wvf['type']} not found.")
