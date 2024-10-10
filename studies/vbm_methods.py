#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm
import rsp.vbm as vbm

################################################################################
# Display each of the VBM noise methods in order of complexity
################################################################################

target = {"range": 0.5e3, "rangeRate": 0.0e3, "rcs": 10}

bw = 5e6

radar = {
    "fcar": 10e9,
    "txPower": 1e3,
    "txGain": 10 ** (30 / 10),
    "rxGain": 10 ** (30 / 10),
    "opTemp": 290,
    "sampRate": 2 * bw,
    "noiseFig": 10 ** (8 / 10),
    "totalLosses": 10 ** (8 / 10),
    "PRF": 500e3,
    "dwell_time": 10e-3,
}

waveform = {"type": "uncoded", "bw": bw}

return_list = [{"type": "memory", "rdot_delta": 1.0e3, "rdot_offset": 0.0e3}]

vbm_name_function_dict = {
    "random phase VBM": vbm._random_phase,
    "uniform bandwidth phase VBM": vbm._uniform_bandwidth_phase,
    "gaussian bandwidth phase VBM": vbm._gaussian_bandwidth_phase,
    "uniform bandwidth phase normalized VBM": vbm._gaussian_bandwidth_phase_normalized,
    "LFM phase VBM": vbm._lfm_phase,
}

for name, func in vbm_name_function_dict.items():
    return_list[0]["vbm_noise_function"] = func
    rdm.gen(target, radar, waveform, return_list, debug=False)
    ax = plt.gca()
    ax.set_title(name)

plt.show()
