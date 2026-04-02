#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp import rdm, Radar, Target, EaPlatform, Return, uncoded_waveform

################################################################################
# Doppler noise is LFM in slow time
################################################################################
# - cleanest to see when target's rangeRate = 0

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

waveform = uncoded_waveform(bw)

return_list = [
    Return(
        target=Target(range=3.5e3, range_rate=0.5e3),
        platform=EaPlatform(
            tx_power=20.0,
            tx_gain=10 ** (5 / 10),
            total_losses=10 ** (3 / 10),
            rdot_delta=2.0e3,
            rdot_offset=0.0e3,
        ),
    )
]

rdm.gen(radar, waveform, return_list, debug=False)

plt.show()
