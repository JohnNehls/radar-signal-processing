#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("..")

from rdm import rdm_gen, plotRDM

################################################################################
# Doppler noise is LFM in slow time
################################################################################
# - cleanest to see when target's rangeRate = 0

tgtInfo = {"range": 0.5e3,
           "rangeRate": 0.0e3,
           "rcs" : 10}

bw = 10e6

radar = {"fcar" : 10e9,
         "txPower": 1e3,
         "txGain" : 10**(30/10),
         "rxGain" : 10**(30/10),
         "opTemp": 290,
         "sampRate": 2*bw,
         "noiseFig": 10**(8/10),
         "totalLosses" : 10**(8/10),
         "PRF": 200e3}

wvf = {"type": "lfm",
       "bw" : bw,
       "T": 1.5e-6,
       'chirpUpDown': 1}

returnInfo = {"type" : "VBM",
              "rdot_delta" : 0.5e3,
              "method" : 2,
              "rdot_offset" : 0.0e3}

dwell_time = 2e-3
Npulses = int(np.ceil(dwell_time* radar ["PRF"]))

rdot_axis, r_axis, total_dc, signal_dc, noise_dc = rdm_gen(tgtInfo, radar,
                                                           wvf, Npulses,
                                                           returnInfo,
                                                           seed=0,
                                                           plotSteps=True)

plotRDM(rdot_axis, r_axis, signal_dc,
        f"SIGNAL: dB doppler processed match filtered {wvf['type']}")
plotRDM(rdot_axis, r_axis, total_dc,
        f"TOTAL: dB doppler processed match filtered {wvf['type']}", cbarRange=False)

plt.show()
