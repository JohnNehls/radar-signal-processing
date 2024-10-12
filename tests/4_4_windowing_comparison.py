#!/usr/bin/env python

import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from rsp.waveform import uncoded_pulse
from rsp.waveform_helpers import plot_pulse_and_spectrum

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

print("##########################")
print("Test windowing")
print("##########################")
# given
fs = 100e6  # sampling frequency in Hz
BW = 11e6
outLength = 1

# make pulse
t_u, mag_u = uncoded_pulse(fs, BW, output_length_T=outLength)

idx = np.where(mag_u != 0)[0].shape[0]

# create windows
chwin = signal.windows.chebwin(mag_u.size, 60)
bhwin = signal.windows.blackmanharris(mag_u.size)
tywin = signal.windows.taylor(mag_u.size)

# apply windows
mag_chwin = chwin * mag_u
mag_bhwin = bhwin * mag_u
mag_tywin = tywin * mag_u

# zero pad
Npad = 4001

print("uncoded")
plot_pulse_and_spectrum(t_u, mag_u, "unfiltered pulse", Npad, spec_dec=True)
print("Chebyshev")
plot_pulse_and_spectrum(t_u, mag_chwin, "chwin filtered pulse", Npad, spec_dec=True)
print("Blackman-Harris")
plot_pulse_and_spectrum(t_u, mag_bhwin, "bhwin filtered pulse", Npad, spec_dec=True)
print("Taylor (should be smaller BW)")
plot_pulse_and_spectrum(t_u, mag_tywin, "tywin filtered pulse", Npad, spec_dec=True)

plt.show(block=BLOCK)
