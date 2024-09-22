#!/usr/bin/env python

import numpy as np
from scipy import signal
import sys

sys.path.append("..")

from rf_datacube import create_dataCube, dopplerProcess_dataCube, R_pf_tgt

# constants
C = 3e8
PI = np.pi

print("##########################")
print("TEST windowing")
print("##########################")
from waveform import makeUncodedPulse
from waveform_helpers import plotPulseAndSpectrum
# given
fs = 100e6 # sampling frequency in Hz
BW = 20e6
outLength = 3


# make pulse
t_u, mag_u = makeUncodedPulse(fs, BW, output_length_T=outLength, centered=True)
print("pulse w/o filter")
plotPulseAndSpectrum(t_u, mag_u, "pulse w/o filter")

# create windows
chwin = signal.windows.chebwin(mag_u.size,60)
bhwin = signal.windows.blackmanharris(mag_u.size)
tywin = signal.windows.taylor(mag_u.size)

print("Chebyshev")
plotPulseAndSpectrum(t_u, chwin*mag_u, "pulse w chwin filter")
print("Blackman-Harris")
plotPulseAndSpectrum(t_u, bhwin*mag_u, "pulse w bhwin filter")
print("Taylor (should be smaller BW)")
plotPulseAndSpectrum(t_u, tywin*mag_u, "pulse w tywin filter")

print("complex tone w/o filter")
## make signal
T = 1/BW*outLength
N = T*fs
t_s = np.arange(N+1)*1/fs
mag_s = np.exp(2j*PI*fs/8*t_s)*mag_u
plotPulseAndSpectrum(t_s, mag_s, "complex tone w/o filter")
plotPulseAndSpectrum(t_s, chwin*mag_s, "complex tone w ch filter")
