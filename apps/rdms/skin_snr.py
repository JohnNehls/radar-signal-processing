#!/usr/bin/env python

import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, Return, lfm_waveform

################################################################################
# skin example
################################################################################

bw = 10e6

radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=2e-3,
)

waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)

return_list = [Return(target=Target(range=3.5e3, range_rate=1.0e3, rcs=10))]

rdm.gen(radar, waveform, return_list, snr=True, debug=True)

plt.show()
