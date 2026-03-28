#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import uncoded_waveform
import rsp.vbm as vbm

################################################################################
# Display each of the VBM noise methods in order of complexity
################################################################################

bw = 5e6

radar: Radar = {
    "fcar": 10e9,
    "txPower": 1e3,
    "txGain": 10 ** (30 / 10),
    "rxGain": 10 ** (30 / 10),
    "opTemp": 290,
    "sampRate": 2 * bw,
    "noiseFactor": 10 ** (8 / 10),
    "totalLosses": 10 ** (8 / 10),
    "PRF": 500e3,
    "dwell_time": 5e-3,
}

waveform = uncoded_waveform(bw)

mem_dict = {"type": "memory",
            "target": {"range": 0.2e3, "rangeRate": 0.0e3},
            "rdot_delta": 1.0e3,
            "rdot_offset": 0.0e3,
            "platform": {
                "txPower": 1.0e3,
                "txGain": 10 ** (5 / 10),
                "totalLosses": 10 ** (3 / 10)}}

vbm_name_function_dict = {
    "random phase VBM": vbm._random_phase,
    "uniform bandwidth phase VBM": vbm._uniform_bandwidth_phase,
    "gaussian bandwidth phase VBM": vbm._gaussian_bandwidth_phase,
    "uniform bandwidth phase normalized VBM": vbm._gaussian_bandwidth_phase_normalized,
    "LFM phase VBM": vbm._lfm_phase,
}

for name, func in vbm_name_function_dict.items():
    mem_dict["vbm_noise_function"] = func
    rdm.gen(radar, waveform, [mem_dict], debug=False)
    ax = plt.gca()
    ax.set_title(name)

print(f"Note:LFM phase VBM is the only method to match the perscribed {mem_dict['rdot_delta']} [m/s] VBM width.")
plt.show()
