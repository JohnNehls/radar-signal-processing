#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from rad_lab import rdm, Radar, Target, EaPlatform, Return
from rad_lab import uncoded_waveform, barker_coded_waveform, random_coded_waveform, lfm_waveform

################################################################################
# Kitchen sink: script showing a sample of all of the options available
################################################################################

bw = 10e6

radar = Radar(
    fcar=10e9,
    tx_power=5e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=2e-3,
)

# Example waveform configurations, the last un-commented on will be used for the RDM
waveform = uncoded_waveform(bw)
waveform = barker_coded_waveform(bw, nchips=13)
waveform = random_coded_waveform(bw, nchips=13)
waveform = lfm_waveform(bw, T=10 / 40e6, chirp_up_down=1)

skin_return = Return(target=Target(range=7.1e3, range_rate=-1e3, rcs=9))
jammer_on_target = Return(
    target=Target(range=3.5e3, range_rate=0.5e3, rcs=10, sv=np.exp(1j * np.pi / 4)),
    platform=EaPlatform(
        tx_power=1,
        tx_gain=10 ** (5 / 10),
        total_losses=10 ** (5 / 10),
        rdot_delta=3.0e3,
        rdot_offset=0.3e3,
        range_offset=-0.2e3,
        delay=1.33e-6,
    ),
)

return_list = [skin_return, jammer_on_target]

rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(
    radar, waveform, return_list, seed=0, plot=True, debug=True, snr=False
)

plt.show()
