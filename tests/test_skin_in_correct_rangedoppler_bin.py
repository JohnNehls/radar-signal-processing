import numpy as np
import pytest
import rsp.pulse_doppler_radar as pdr
from rsp import rdm
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import uncoded_waveform, barker_waveform, random_waveform, lfm_waveform

BW = 10e6

RADAR: Radar = {
    "fcar": 10e9,
    "txPower": 1e3,
    "txGain": 10 ** (30 / 10),
    "rxGain": 10 ** (30 / 10),
    "opTemp": 290,
    "sampRate": 2 * BW,
    "noiseFactor": 10 ** (8 / 10),
    "totalLosses": 10 ** (8 / 10),
    "PRF": 50e3,
    "dwell_time": 2e-3,
}

RETURN = {"type": "skin", "target": {"range": 8.4e3, "rangeRate": 3.2e3, "rcs": 10}}

WAVEFORMS = [
    uncoded_waveform(BW),
    barker_waveform(BW, nchips=5),
    barker_waveform(BW, nchips=13),
    random_waveform(BW, nchips=13),
    lfm_waveform(BW, T=10 / 40e6, chirpUpDown=1),
]


def check_max_in_expected_bin(waveform):
    rdot_axis, r_axis, _total_dc, signal_dc = rdm.gen(RADAR, waveform, [RETURN], plot=False)

    range_expected = pdr.range_aliased(RETURN["target"]["range"], RADAR["PRF"])
    rangeRate_expected = pdr.rangeRate_aliased_prf_f0(
        RETURN["target"]["rangeRate"], RADAR["PRF"], RADAR["fcar"]
    )
    i = np.argmin(abs(r_axis - range_expected))
    j = np.argmin(abs(rdot_axis - rangeRate_expected))

    max_index_flat = np.argmax(abs(signal_dc))
    max_range_index, max_rdot_index = np.unravel_index(max_index_flat, signal_dc.shape)

    assert max_range_index == i, f"Range bin mismatch: got {max_range_index}, expected {i}"
    assert max_rdot_index == j, f"Doppler bin mismatch: got {max_rdot_index}, expected {j}"


@pytest.mark.parametrize("waveform", WAVEFORMS, ids=lambda w: w["type"] + str(w.get("nchips", "")))
def test_skin_return_in_correct_bin(waveform):
    check_max_in_expected_bin(waveform)
