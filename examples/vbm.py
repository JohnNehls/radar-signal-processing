#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm

################################################################################
# Doppler noise is LFM in slow time
################################################################################
# - cleanest to see when target's rangeRate = 0

target = {"range": 0.5e3, "rangeRate": 0.0e3, "rcs": 10}

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

waveform = {"type": "uncoded", "bw": bw}

return_list = [
    {
        "type": "memory",
        "rdot_delta": 2.0e3,
        "rdot_offset": 0.0e3,
        "platform": {
            "txPower": 200.0e3,
            "txGain": 10 ** (30 / 10),
            "totalLosses": 10 ** (3 / 10),
        },
    }
]

rdm.gen(target, radar, waveform, return_list, debug=False)

plt.show()
