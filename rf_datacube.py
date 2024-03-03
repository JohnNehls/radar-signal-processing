#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from scipy import fft
from scipy import signal
import matplotlib.pyplot as plt

# constants
C = 3e8
PI = np.pi

def create_dataCube(fs, PRF, Np):
    """data cube
    ouputs unprocessed datacube, both in fast and slow time
    """
    Nr = round(fs/PRF)
    dc = np.zeros((Nr,Np),dtype=np.complex64)

    return dc

def dopplerProcess_dataCube(dc, fs, PRF):
    """data cube
    ouputs:\n
    dataCube : \n
    f_axis : [-fs/2, fs/2)\n
    r_axis : [delta_r, R_ambigious]\n
    """

    dR_grid = C/(2*fs)
    Nr = round(fs/PRF)

    R_axis = np.arange(1,Nr+1)*dR_grid # Process fast time
    f_axis = fft.fftshift(fft.fftfreq(Np,1/fs)) # process slow time

    for i in range(dc.shape[0]):
        dc[i,:] = fft.fftshift(fft.fft(dc[i]))

    return dc, f_axis, R_axis

def R_pf_tgt(pf_pos : list, pf_vel : list, tgt_pos : list, tgt_vel : list):

    R_vec = np.array([tgt_pos[0] - pf_pos[0], tgt_pos[1] - pf_pos[1], tgt_pos[2] - pf_pos[2]])
    R_unit_vec = R_vec/norm(R_vec)

    R_mag = np.sqrt(R_vec[0]**2 + R_vec[1]**2 + R_vec[2]**2)

    R_dot = np.dot(tgt_vel, R_unit_vec) - np.dot(pf_vel, R_unit_vec)

    return R_vec, R_mag, R_dot


plt.close('all')

## TEST datacube processing
# given
fs = 20e6 # sampling frequency in Hz
PRF = 100e3 # Hz
Np = 256 # number of pulses

# calc
dc = create_dataCube(fs, PRF, Np)
dtPulse = 1/PRF
t_ar = np.arange(Np)*dtPulse
dc[98] = np.exp(2j*PI*PRF/4*t_ar)

fig, ax = plt.subplots(1,2)
fig.suptitle("test datacube processing")
ax[0].set_title("unprocessed datacube")
ax[0].imshow(abs(dc), origin='lower')
ax[0].set_xlabel("slow time [PRI]")
ax[0].set_ylabel("fast time [fs]")

dcp, f_ax, r_ax =dopplerProcess_dataCube(dc, fs, PRF)

ax[1].set_title("processed datacube")
ax[1].pcolormesh(f_ax*1e-6, r_ax, abs(dcp))
ax[1].set_xlabel("frequency [MHz]")
ax[1].set_ylabel("range [m]")
plt.tight_layout()

## TEST Windowing
from waveforms import makeUncodedPulse
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


print("PROBLEM 6")
R_pf_tgt([0, 0, 3048], [300,0,0], [5e3,0,3048], [-300,0,0])
