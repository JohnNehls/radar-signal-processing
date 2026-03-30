#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import lfm_waveform
from rsp.returns import Target, EaPlatform, Return

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

waveform = lfm_waveform(bw, T=1.5e-6, chirpUpDown=1)

return_list = [
    Return(
        target=Target(range=0.5e3, rangeRate=0.0e3),
        platform=EaPlatform(
            txPower=2.0, txGain=10 ** (5 / 10), totalLosses=10 ** (3 / 10),
            rdot_delta=0.5e3, rdot_offset=0.0e3,
        ),
    )
]

rdm.gen(radar, waveform, return_list, debug=False)

plt.show()
