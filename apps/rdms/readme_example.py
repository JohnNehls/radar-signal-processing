#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm, Radar, Target, Return, barker_coded_waveform

radar = Radar(
    fcar=10e9,
    txPower=1e3,
    txGain=10 ** (30 / 10),
    rxGain=10 ** (30 / 10),
    opTemp=290,
    sampRate=20e6,
    noiseFactor=10 ** (8 / 10),
    totalLosses=10 ** (8 / 10),
    PRF=200e3,
    dwell_time=2e-3,
)

waveform = barker_coded_waveform(10e6, nchips=13)

return_list = [Return(target=Target(range=0.5e3, rangeRate=1.0e3, rcs=1))]

rdm.gen(radar, waveform, return_list)

plt.show()
