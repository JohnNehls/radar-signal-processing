#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm
from rsp.rdm_helpers import plotRDM

################################################################################
# Doppler noise is LFM in slow time
################################################################################
# - cleanest to see when target's rangeRate = 0

tgt = {"range": 0.5e3, "rangeRate": 0.0e3, "rcs": 10}

bw = 10e6

radar = {
    "fcar": 10e9,
    "txPower": 1e3,
    "txGain": 10 ** (30 / 10),
    "rxGain": 10 ** (30 / 10),
    "opTemp": 290,
    "sampRate": 2 * bw,
    "noiseFig": 10 ** (8 / 10),
    "totalLosses": 10 ** (8 / 10),
    "PRF": 200e3,
    "dwell_time": 2e-3,
}

wvf = {"type": "lfm", "bw": bw, "T": 1.5e-6, "chirpUpDown": 1}

return_list = [{"type": "memory", "rdot_delta": 0.5e3, "method": 2, "rdot_offset": 0.0e3}]

rdot_axis, r_axis, total_dc, _, _ = rdm.rdm_gen(
    tgt, radar, wvf, return_list, seed=0, plotSteps=True
)

plotRDM(rdot_axis, r_axis, total_dc, f"Total RDM for {wvf['type']}")

plt.show()
