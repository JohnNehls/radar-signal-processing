#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm
from rsp.rdm_helpers import plotRDM

################################################################################
# Kitchen sink: script showing a sample of all of the options available
################################################################################

bw = 10e6

tgt = {"range": 3.5e3, "rangeRate": 0.5e3, "rcs": 10}

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

wvf = {"type": None}  # noise test

wvf = {"type": "uncoded", "bw": bw}

wvf = {"type": "barker", "nchips": 13, "bw": bw}

wvf = {"type": "random", "nchips": 13, "bw": bw}

wvf = {"type": "lfm", "bw": bw, "T": 10 / 40e6, "chirpUpDown": 1}

return_list = [
    {
        "type": "memory",
        "rdot_delta": 0.5e3,
        "rdot_offset": 0.2e3,
        "range_offset": -0.2e3,
    },
    {"type": "skin"},
]

rdot_axis, r_axis, total_dc, _, _ = rdm.rdm_gen(
    tgt, radar, wvf, return_list, seed=0, plotSteps=True
)

plotRDM(rdot_axis, r_axis, total_dc, f"Total RDM for {wvf['type']}")

plt.show()
