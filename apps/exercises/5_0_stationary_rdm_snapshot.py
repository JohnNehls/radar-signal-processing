#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import lfm_waveform
from rsp.returns import Target, Return

################################################################################
# skin example
################################################################################
# - Matlab solution is incorrect due to not accounting for time-bandwidth prod in SNR
# - Matlab solution neglects range walk off, our SNR may be off when rangeRate>>0


bw = 10e6

radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    samp_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=2e-3,
)

waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)

return_list = [Return(target=Target(range=3.5e3, range_rate=0.0e3, rcs=10))]

rdm.gen(radar, waveform, return_list, snr=True, debug=True)

plt.show()
