#!/usr/bin/env python
"""Skin return with SNR verification.

Generate an RDM for a single moving target using an LFM waveform and compare
the measured SNR in the RDM to the predicted value from the range equation.

The `snr=True` flag prints the range-equation SNR and the measured peak SNR
in the RDM. The `debug=True` flag shows intermediate processing steps.
"""

import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, Return, lfm_waveform

bw = 10e6  # waveform bandwidth [Hz]

# -- Radar system --
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,  # Nyquist rate for the waveform
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=2e-3,
)

# -- LFM waveform: 1 us up-chirp --
waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)

# -- Target at 3.5 km, closing at 1 km/s, 10 m^2 RCS --
return_list = [Return(target=Target(range=3.5e3, range_rate=1.0e3, rcs=10))]

# -- Generate RDM with SNR check and debug plots --
rdm.gen(radar, waveform, return_list, snr=True, debug=True)

plt.show()
