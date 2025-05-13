#!/usr/bin/env python

import numpy as np

from rsp import rdm
import rsp.pulse_doppler_radar as pdr
################################################################################
# Test to make sure the return is where we expect it
# - if rangeRate is too large, the checks will fail due to the return walking over the CPI
################################################################################

bw = 10e6

radar = {
    "fcar": 10e9,
    "txPower": 1e3,
    "txGain": 10 ** (30 / 10),
    "rxGain": 10 ** (30 / 10),
    "opTemp": 290,
    "sampRate": 2 * bw,
    "noiseFactor": 10 ** (8 / 10),
    "totalLosses": 10 ** (8 / 10),
    "PRF": 50e3,
    "dwell_time": 2e-3,
}

return_list = [{"type": "skin", "target": {"range": 8.4e3, "rangeRate": 3.2e3, "rcs": 10}}]


def compare_max_indices(signal_dc, return_item):
    # expected
    range_expected = pdr.range_aliased(return_item["target"]["range"], radar["PRF"])
    rangeRate_expected = pdr.rangeRate_aliased_prf_f0(
        return_item["target"]["rangeRate"], radar["PRF"], radar["fcar"]
    )
    i = np.argmin(abs(r_axis - range_expected))
    j = np.argmin(abs(rdot_axis - rangeRate_expected))
    # find the max indices
    max_index_flat = np.argmax(abs(signal_dc))
    # convert the flattened index to row and column indices
    max_range_index, max_rdot_index = np.unravel_index(max_index_flat, signal_dc.shape)
    # compare to expected
    print(f"\t{max_range_index == i=}")
    print(f"\t\t{max_range_index=} {i=}")
    print(f"\t{max_rdot_index == j=}")
    print(f"\t\t{max_rdot_index=} {j=}")


print("uncoded")
waveform = {"type": "uncoded", "bw": bw}
rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(radar, waveform, return_list, plot=True)
compare_max_indices(signal_dc, return_list[0])

print("barker5")
waveform = {"type": "barker", "nchips": 5, "bw": bw}
rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(radar, waveform, return_list, plot=False)
compare_max_indices(signal_dc, return_list[0])

print("barker13")
waveform = {"type": "barker", "nchips": 13, "bw": bw}
rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(radar, waveform, return_list, plot=False)
compare_max_indices(signal_dc, return_list[0])

print("random13")
waveform = {"type": "random", "nchips": 13, "bw": bw}
rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(radar, waveform, return_list, plot=False)
compare_max_indices(signal_dc, return_list[0])

print("lfm")
waveform = {"type": "lfm", "bw": bw, "T": 10 / 40e6, "chirpUpDown": 1}
rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(radar, waveform, return_list, plot=False)
compare_max_indices(signal_dc, return_list[0])
