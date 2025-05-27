#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
from rsp import rdm
from skimage.feature import peak_local_max
from scipy.ndimage import convolve
from rsp.rdm_detector import gaussian_kernel

################################################################################
# skin example
################################################################################
# - Matlab solution is incorrect due to not accounting for time-bandwidth prod in SNR
# - Matlab solution neglects range walk off, our SNR may be off when rangeRate>>0

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

bw = 10e6

radar = {
    "fcar": 10e9,
    "txPower": 1e3,
    "txGain": 10 ** (30 / 10),
    "rxGain": 10 ** (30 / 10),
    "opTemp": 290,
    "sampRate": 2 * bw,
    "noiseFactor": 10 ** (8 / 10),
    "totalLosses": 10 ** (8 / 10),
    "PRF": 50e3,
    "dwell_time": 2e-3,
}

waveform = {"type": "barker", "nchips": 5, "bw": bw}

return_list = [
    {
        "type": "skin",
        "target": {"range": 1.5e3, "rangeRate": 0.0e3, "rcs": 10},
    },
    {"type": "skin", "target": {"range": 2.4e3, "rangeRate": 0.2e3, "rcs": 1}},
]

rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(
    radar, waveform, return_list, snr=False, debug=False, plot=False
)

min_d = 20
data = abs(total_dc)
peaks = peak_local_max(data, min_distance=min_d)
print(f"{peaks=}")
print(f"{len(peaks)=}")

kernel = gaussian_kernel(9, 3)
smoothed_data = convolve(abs(total_dc), kernel)
speaks = peak_local_max(smoothed_data, min_distance=min_d)
print(f"{speaks=}")
print(f"{len(speaks)=}")
for p in speaks:
    print(f"{data[p[0],p[1]]}")

raise Exception("This funcionality/test is incomplete")

plt.show(block=BLOCK)
