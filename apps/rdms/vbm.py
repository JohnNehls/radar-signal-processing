#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import uncoded_waveform

################################################################################
# Doppler noise is LFM in slow time
################################################################################
# - cleanest to see when target's rangeRate = 0

bw = 10e6

radar = Radar(
    fcar=10e9,
    txPower=1e3,
    txGain=10 ** (30 / 10),
    rxGain=10 ** (30 / 10),
    opTemp=290,
    sampRate=2 * bw,
    noiseFactor=10 ** (8 / 10),
    totalLosses=10 ** (8 / 10),
    PRF=200e3,
    dwell_time=2e-3,
)

waveform = uncoded_waveform(bw)

return_list = [
    {
        "type": "memory",
        "target": {"range": 3.5e3, "rangeRate": 0.5e3},
        "rdot_delta": 2.0e3,
        "rdot_offset": 0.0e3,
        "platform": {
            "txPower": 20.0,
            "txGain": 10 ** (5 / 10),
            "totalLosses": 10 ** (3 / 10),
        },
    }
]

rdm.gen(radar, waveform, return_list, debug=False)

plt.show()
