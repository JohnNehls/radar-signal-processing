#!/usr/bin/env python
"""VBM electronic attack with an LFM waveform.

Same concept as vbm.py but using an LFM (chirp) waveform instead of uncoded.
LFM provides better range resolution via pulse compression, so the VBM
Doppler noise band is spread across fewer range bins but the same Doppler
extent.

The target is stationary (range_rate=0) to isolate the VBM effect in the
Doppler dimension.
"""

import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, EaPlatform, Return, lfm_waveform

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

# -- LFM waveform: 1.5 us up-chirp --
waveform = lfm_waveform(bw, T=1.5e-6, chirp_up_down=1)

# -- Stationary target with VBM jammer (±500 m/s Doppler spread) --
return_list = [
    Return(
        target=Target(range=0.5e3, range_rate=0.0e3),
        platform=EaPlatform(
            tx_power=2.0,
            tx_gain=10 ** (5 / 10),
            total_losses=10 ** (3 / 10),
            rdot_delta=0.5e3,  # narrow VBM spread [m/s]
            rdot_offset=0.0e3,
        ),
    )
]

rdm.gen(radar, waveform, return_list, debug=False)

plt.show()
