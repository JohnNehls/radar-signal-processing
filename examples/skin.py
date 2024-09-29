#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from rdm import rdm_gen, plotRDM

################################################################################
# skin example
################################################################################
# - Matlab solution is incorrect due to not accounting for time-bandwidth prod in SNR
# - Matlab solution neglects range walk off, our SNR may be off when rangeRate>>0

tgtInfo = {"range": 3.5e3,
           "rangeRate": 0.5e3,
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
         "PRF": 200e3,
         "dwell_time" : 2e-3}

wvf = {"type": "lfm",
       "bw" : bw,
       "T": 1.0e-6,
       'chirpUpDown': 1}

# wvf = {"type": "uncoded",
#        "bw" : bw}


returnInfo_list = [{"type" : "skin"}]

rdot_axis, r_axis, total_dc, signal_dc, noise_dc = rdm_gen(tgtInfo, radar, wvf,
                                                           returnInfo_list,
                                                           seed=0,
                                                           plotSteps=True)

plotRDM(rdot_axis, r_axis, signal_dc,
        f"SIGNAL: dB doppler processed match filtered {wvf['type']}")
plotRDM(rdot_axis, r_axis, total_dc, f"TOTAL: dB doppler processed match filtered {wvf['type']}", cbarRange=False)

plt.show()
