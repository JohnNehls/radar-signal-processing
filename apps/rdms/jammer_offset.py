#!/usr/bin/env python
"""Jammer with range and Doppler offset.

Demonstrate a DRFM jammer that retransmits with offsets in both range and
Doppler, pulling the apparent target away from its true position.

EaPlatform offset parameters:
  - range_offset: shifts the jammer response in range [m] (negative = closer).
  - rdot_offset: shifts the VBM noise center in Doppler [m/s].
  - rdot_delta: width of the VBM Doppler spread [m/s].
  - delay: additional time delay before retransmission [s].
"""

import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, Return, EaPlatform, lfm_waveform

# -- Radar --
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=20e6,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=3e-3,
)

waveform = lfm_waveform(bw=10e6, T=1.0e-6, chirp_up_down=1)

# -- Target at 3.5 km with a jammer that offsets in range and Doppler --
return_list = [
    Return(
        target=Target(range=3.5e3, range_rate=1e3, rcs=10),
        platform=EaPlatform(
            tx_power=0.5,  # jammer power [W]
            tx_gain=10 ** (3 / 10),  # jammer gain [linear], 3 dB
            total_losses=10 ** (5 / 10),  # jammer losses [linear], 5 dB
            rdot_delta=2.0e3,  # VBM Doppler spread [m/s]
            rdot_offset=-1e3,  # shift VBM center -1 km/s from target
            range_offset=-0.2e3,  # shift jammer 200 m closer than target
            delay=0,  # no additional retransmission delay [s]
        ),
    )
]

rdm.gen(radar, waveform, return_list)
plt.show()
