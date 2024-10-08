#!/usr/bin/env python

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from rsp.constants import PI
from rsp.waveform import uncoded_pulse
from rsp.waveform_helpers import plot_pulse_and_spectrum

print("##########################")
print("TEST windowing")
print("##########################")
# given
fs = 100e6  # sampling frequency in Hz
BW = 20e6
outLength = 3


# make pulse
t_u, mag_u = uncoded_pulse(fs, BW, output_length_T=outLength, centered=True)
print("pulse w/o filter")
plot_pulse_and_spectrum(t_u, mag_u, "pulse w/o filter")

# create windows
chwin = signal.windows.chebwin(mag_u.size, 60)
bhwin = signal.windows.blackmanharris(mag_u.size)
tywin = signal.windows.taylor(mag_u.size)

print("Chebyshev")
plot_pulse_and_spectrum(t_u, chwin * mag_u, "pulse w chwin filter")
print("Blackman-Harris")
plot_pulse_and_spectrum(t_u, bhwin * mag_u, "pulse w bhwin filter")
print("Taylor (should be smaller BW)")
plot_pulse_and_spectrum(t_u, tywin * mag_u, "pulse w tywin filter")

print("complex tone w/o filter")
## make signal
T = 1 / BW * outLength
N = T * fs
t_s = np.arange(N + 1) * 1 / fs
mag_s = np.exp(2j * PI * fs / 8 * t_s) * mag_u
plot_pulse_and_spectrum(t_s, mag_s, "complex tone w/o filter")
plot_pulse_and_spectrum(t_s, chwin * mag_s, "complex tone w ch filter")

plt.show()
