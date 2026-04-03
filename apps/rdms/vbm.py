#!/usr/bin/env python
"""Velocity-Bin Masking (VBM) electronic attack example.

VBM is a DRFM jamming technique that spreads energy across multiple Doppler
bins to mask the true target velocity. The jammer modulates the phase of
the retransmitted signal in slow time, creating a band of Doppler noise
centered on the target's Doppler cell.

Parameters:
  - rdot_delta: width of the Doppler spread [m/s] — controls how many
    velocity bins are contaminated.
  - rdot_offset: offset of the noise band center from the target [m/s].

The VBM noise appears as LFM in slow time. It is cleanest to observe
when the target's range-rate is 0.
"""

import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, EaPlatform, Return, uncoded_waveform

bw = 10e6  # waveform bandwidth [Hz]

# -- Radar --
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

# -- Target with VBM jammer --
# The Return combines a skin target with an EaPlatform (electronic attack).
# The jammer spreads Doppler energy over ±2 km/s around the target.
return_list = [
    Return(
        target=Target(range=3.5e3, range_rate=0.5e3),
        platform=EaPlatform(
            tx_power=20.0,  # jammer transmit power [W]
            tx_gain=10 ** (5 / 10),  # jammer antenna gain [linear], 5 dB
            total_losses=10 ** (3 / 10),  # jammer losses [linear], 3 dB
            rdot_delta=2.0e3,  # Doppler spread width [m/s]
            rdot_offset=0.0e3,  # Doppler offset from target [m/s]
        ),
    )
]

rdm.gen(radar, waveform, return_list, debug=False)

plt.show()
