#!/usr/bin/env python

import numpy as np

from rsp import rdm

################################################################################
# Make sure VBM is were we expect
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

return_list = [
    {
        "type": "memory",
        "target": {"range": 1.5e3, "rangeRate": 0.2e3},
        "rdot_delta": 0.1e3,
        "rdot_offset": 0.0e3,
        "range_offset": 0.0e3,
        "delay": 0.0e-6,
        "platform": {
            "txPower": 5.0e3,
            "txGain": 10 ** (30 / 10),
            "totalLosses": 10 ** (3 / 10),
        },
    }
]


def compare_max_indices(signal_dc, return_item):
    max_index_flat = np.argmax(abs(signal_dc))
    # Convert the flattened index to row and column indices
    max_range_index, max_rdot_index = np.unravel_index(max_index_flat, signal_dc.shape)

    i = np.argmin(abs(r_axis - return_item["target"]["range"]))
    j = np.argmin(abs(rdot_axis - return_item["target"]["rangeRate"]))
    print(f"\t{max_range_index == i=}")
    print(f"\t{max_rdot_index == j=}")


print("uncoded")
waveform = {"type": "uncoded", "bw": bw}
rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(radar, waveform, return_list, plot=False)
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
