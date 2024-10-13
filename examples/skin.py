#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm

################################################################################
# skin example
################################################################################
# - Matlab solution is incorrect due to not accounting for time-bandwidth prod in SNR
# - Matlab solution neglects range walk off, our SNR may be off when rangeRate>>0

target = {"range": 3.5e3, "rangeRate": 0.0e3, "rcs": 10}

bw = 10e6

radar = {
    "fcar": 10e9,
    "txPower": 10e3,
    "txGain": 10 ** (30 / 10),
    "rxGain": 10 ** (30 / 10),
    "opTemp": 290,
    "sampRate": 2 * bw,
    "noiseFactor": 10 ** (8 / 10),
    "totalLosses": 10 ** (8 / 10),
    "PRF": 200e3,
    "dwell_time": 2e-3,
}

waveform = {"type": "lfm", "bw": bw, "T": 1.0e-6, "chirpUpDown": 1}

return_list = [{"type": "skin"}]

rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(target, radar, waveform, return_list)

# plt.figure()
# p = plt.pcolormesh(rdot_axis, r_axis, 20 * np.log10(abs(total_dc)/np.sqrt(radar["txPower"] * c.RADAR_LOAD)))
# cbar = plt.colorbar(p)
# cbar.set_label("Power [dB]")

# plt.figure()
# p = plt.pcolormesh(rdot_axis, r_axis, 20 * np.log10(abs(total_dc)/np.sqrt(1e-3 * c.RADAR_LOAD)))
# cbar = plt.colorbar(p)
# cbar.set_label("Power [dBm]")

plt.show()
