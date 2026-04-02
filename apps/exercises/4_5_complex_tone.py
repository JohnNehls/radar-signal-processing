#!/usr/bin/env python

from scipy import signal
import matplotlib.pyplot as plt
from rad_lab.waveform import complex_tone_pulse
from rad_lab.waveform_helpers import plot_pulse_and_spectrum


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
plot_pulse_and_spectrum(t_s, mag_s, "complex tone without filter", n_pad=1024)
plot_pulse_and_spectrum(t_s, mag_chwin_s, "complex tone with chwin filter", n_pad=1024)

plt.show()
