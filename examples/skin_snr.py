#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm

################################################################################
# skin example
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

waveform = {"type": "lfm", "bw": bw, "T": 1.0e-6, "chirpUpDown": 1}

return_list = [{"type": "skin",
                "target": {"range": 3.5e3, "rangeRate": 1.0e3, "rcs": 10}}
               ]

rdm.gen(radar, waveform, return_list, snr=True, debug=True)

plt.show()
