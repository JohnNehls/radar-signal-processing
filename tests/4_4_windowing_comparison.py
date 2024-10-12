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

mag_pulse = np.append(mag_u, np.zeros(Npad))
mag_chwin = np.append(mag_chwin, np.zeros(Npad))
mag_bhwin = np.append(mag_bhwin, np.zeros(Npad))
mag_tywin = np.append(mag_tywin, np.zeros(Npad))
dt = t_u[1] - t_u[0]
t_pulse = np.append(t_u, np.arange(Npad) * dt + t_u[-1] + dt)

print("uncoded")
plot_pulse_and_spectrum(t_pulse, mag_pulse, "unfiltered pulse", spec_dec=True)
print("Chebyshev")
plot_pulse_and_spectrum(t_pulse, mag_chwin, "chwin filtered pulse", spec_dec=True)
print("Blackman-Harris")
plot_pulse_and_spectrum(t_pulse, mag_bhwin, "bhwin filtered pulse", spec_dec=True)
print("Taylor (should be smaller BW)")
plot_pulse_and_spectrum(t_pulse, mag_tywin, "tywin filtered pulse", spec_dec=True)

plt.show(block=BLOCK)
