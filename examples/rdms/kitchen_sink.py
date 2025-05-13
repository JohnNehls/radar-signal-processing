#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm
import numpy as np

################################################################################
# Kitchen sink: script showing a sample of all of the options available
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
    "PRF": 200e3,
    "dwell_time": 2e-3,
}

# Example waveform configurations, the last un-commented on will be used for the RDM
waveform = {"type": None}  # noise test
waveform = {"type": "uncoded", "bw": bw}
waveform = {"type": "barker", "nchips": 13, "bw": bw}
waveform = {"type": "random", "nchips": 13, "bw": bw}
waveform = {"type": "lfm", "bw": bw, "T": 10 / 40e6, "chirpUpDown": 1}

return_list = [
    {
        "type": "memory",
        "target": {"range": 3.5e3, "rangeRate": 0.5e3, "sv": np.exp(1j * np.pi / 4)},
        "rdot_delta": 3.0e3,
        "rdot_offset": 0.3e3,
        "range_offset": -0.2e3,
        "delay": 1.33e-6,
        "platform": {
            "txPower": 5.0e3,
            "txGain": 10 ** (30 / 10),
            "totalLosses": 10 ** (3 / 10),
        },
    },
    {"type": "skin", "target": {"range": 3.5e3, "rangeRate": 0.5e3, "rcs": 10}},
]

rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(
    radar, waveform, return_list, seed=0, plot=True, debug=False, snr=False
)

plt.show()
