#!/usr/bin/env python

import sys
from scipy import signal
import matplotlib.pyplot as plt
from rsp.waveform import complex_tone_pulse
from rsp.waveform_helpers import plot_pulse_and_spectrum

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

print("complex tone without filter")
fs = 200e6  # sampling frequency in Hz
BW = 11e6

## make signal
t_s, mag_s = complex_tone_pulse(fs, BW, fs / 8)

# create window
chwin = signal.windows.chebwin(mag_s.size, 60)

# apply windows
mag_chwin_s = chwin * mag_s

# plot
plot_pulse_and_spectrum(t_s, mag_s, "complex tone without filter", Npad=1024)
plot_pulse_and_spectrum(t_s, mag_chwin_s, "complex tone with chwin filter", Npad=1024)

plt.show(block=BLOCK)
