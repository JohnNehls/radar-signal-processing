#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import uncoded_waveform, barker_waveform, random_waveform, lfm_waveform
import numpy as np

################################################################################
# Kitchen sink: script showing a sample of all of the options available
################################################################################

bw = 10e6

radar: Radar = {
    "fcar": 10e9,
    "txPower": 5e3,
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
waveform = uncoded_waveform(bw)
waveform = barker_waveform(bw, nchips=13)
waveform = random_waveform(bw, nchips=13)
waveform = lfm_waveform(bw, T=10 / 40e6, chirpUpDown=1)

target_dict = {"type": "skin", "target": {"range": 3.5e3,
                                          "rangeRate": 0.5e3,
                                          "rcs": 10}}
mem_on_target_dict = {"type": "memory",
                      "target": {"range": target_dict['target']['range'],
                                 "rangeRate": target_dict['target']['rangeRate'],
                                 "sv": np.exp(1j * np.pi / 4)},
                      "rdot_delta": 3.0e3,
                      "rdot_offset": 0.3e3,
                      "range_offset": -0.2e3,
                      "delay": 1.33e-6,
                      "platform": {
                          "txPower": 1,
                          "txGain": 10 ** (5 / 10),
                          "totalLosses": 10 ** (5 / 10)}}

return_list = [ target_dict, mem_on_target_dict]

rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(
    radar, waveform, return_list, seed=0, plot=True, debug=True, snr=False
)

plt.show()
