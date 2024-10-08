import numpy as np
from numpy.linalg import norm
from scipy import fft
from . import constants as c

BARKER_DICT = {
    2: [1, -1],  # could also be [ 1, 1]
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def uncoded_pulse(sampleRate, BW, output_length_T=1, t_start=0, normalize=True, centered=False):
    """baseband uncoded pulse"""
    assert output_length_T >= 1, "Error: must output a full pulse"
    assert sampleRate / BW >= 2, "Error: sample rate below Nyquist"

    T = 1 / BW
    dt = 1 / sampleRate
    t = np.arange(t_start, t_start + output_length_T * T + dt, dt)
    mag = np.zeros(t.size)

    mag[np.where((t >= t_start) & (t <= T + t_start))] = 1

    if normalize:
        mag = mag / norm(mag)

    if centered:
        mag = fft.fftshift(mag)

    return t, mag


def coded_pulse(sampleRate, BW, code, output_length_T=1, t_start=0, normalize=True, centered=False):
    """baseband coded pulse"""
    assert output_length_T >= 1, "Error: must output a full pulse"

    nChips = len(code)

    Tc = 1 / BW
    T = nChips * Tc
    dt = 1 / sampleRate

    samplesPerChip = round(Tc * sampleRate)

    mag = np.zeros((nChips, samplesPerChip))

    for i, val in enumerate(code):
        mag[i, :] = val

    mag = mag.flatten()

    if output_length_T > 1:
        tmp = np.zeros(round(output_length_T * T * sampleRate))
        tmp[: mag.size] = mag
        mag = tmp

    t = np.arange(mag.size) * dt + t_start

    if normalize:
        mag = mag / norm(mag)

    if centered:
        mag = fft.fftshift(mag)

    return t, mag


def barker_coded_pulse(sampleRate, BW, nChips, output_length_T=1, t_start=0, normalize=True):
    """baseband Barker coded pulse"""
    assert nChips in BARKER_DICT, f"Error: {nChips=} is not a valid Barker code."
    assert nChips == len(BARKER_DICT[nChips]), "Error: Barker dict is incorrect"
    return coded_pulse(
        sampleRate,
        BW,
        BARKER_DICT[nChips],
        output_length_T=output_length_T,
        t_start=t_start,
        normalize=normalize,
    )


def random_coded_pulse(sampleRate, BW, nChips, output_length_T=1, t_start=0, normalize=True):
    """baseband random bi-phase coded pulse"""
    code_rand = np.random.choice([1, -1], size=nChips)
    return coded_pulse(
        sampleRate,
        BW,
        code_rand,
        output_length_T=output_length_T,
        t_start=t_start,
        normalize=normalize,
    )


def lfm_pulse(sampleRate, BW, T, chirpUpDown, output_length_T=1, t_start=0, normalize=True):
    """baseband LFM pulse"""
    assert output_length_T >= 1, "Error: must output a full pulse"
    dt = 1 / sampleRate
    t = np.arange(t_start, t_start + output_length_T * T + dt, dt)
    f = chirpUpDown * (-BW * t + BW * t**2 / T) / 2
    mag = np.zeros(t.size, dtype=np.complex64)
    i_ar = np.where((t >= t_start) & (t <= (t_start + T)))
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
