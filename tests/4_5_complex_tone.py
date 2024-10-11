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

    print("complex tone without filter")
fs = 100e6  # sampling frequency in Hz
BW = 11e6
outLength = 1

t_u, mag_u = uncoded_pulse(fs, BW, output_length_T=outLength, centered=False)

## make signal
T = 1 / BW * outLength
N = T * fs
t_s = np.arange(N + 1) * 1 / fs
mag_s = np.exp(2j * PI * fs / 8 * t_s) * mag_u

# create window
chwin = signal.windows.chebwin(mag_u.size, 60)

# apply windows
mag_chwin = chwin * mag_s

#zero pad
Npad = 48
mag_pulse = np.append(mag_s, np.zeros(Npad))
mag_chwin = np.append(mag_chwin, np.zeros(Npad))
dt = t_u[1] - t_u[0]
t_pulse = np.append(t_u, np.arange(Npad)*dt +t_u[-1]+dt)


plot_pulse_and_spectrum(t_pulse, mag_pulse, "complex tone without filter")
plot_pulse_and_spectrum(t_pulse, mag_chwin, "complex tone with chwin filter")

plt.show(block=BLOCK)
