#!/usr/bin/env python

import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from rsp.constants import PI
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
BW = 20e6
outLength = 3


# make pulse
t_u, mag_u = uncoded_pulse(fs, BW, output_length_T=outLength, centered=True)
print("unfiltered pulse")
plot_pulse_and_spectrum(t_u, mag_u, "unfiltered pulse")

# create windows
chwin = signal.windows.chebwin(mag_u.size, 60)
bhwin = signal.windows.blackmanharris(mag_u.size)
tywin = signal.windows.taylor(mag_u.size)

print("Chebyshev")
plot_pulse_and_spectrum(t_u, chwin * mag_u, "chwin filtered pulse")
print("Blackman-Harris")
plot_pulse_and_spectrum(t_u, bhwin * mag_u, "bhwin filtered pulse")
print("Taylor (should be smaller BW)")
plot_pulse_and_spectrum(t_u, tywin * mag_u, "tywin filtered pulse")

print("complex tone without filter")
## make signal
T = 1 / BW * outLength
N = T * fs
t_s = np.arange(N + 1) * 1 / fs
mag_s = np.exp(2j * PI * fs / 8 * t_s) * mag_u
plot_pulse_and_spectrum(t_s, mag_s, "complex tone without filter")
plot_pulse_and_spectrum(t_s, chwin * mag_s, "complex tone with chwin filter")

plt.show(block=BLOCK)
