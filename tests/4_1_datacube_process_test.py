#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from rsp.rf_datacube import create_dataCube, dopplerProcess_dataCube
from rsp.constants import C, PI

print("##########################")
print("TEST datacube processing")
print("##########################")

# given
fs = 20e6 # sampling frequency in Hz
PRF = 100e3 # Hz
Np = 256 # number of pulses

# calc
dc = create_dataCube(fs, PRF, Np)
dtPulse = 1/PRF
t_ar = np.arange(Np)*dtPulse
dc[98,:] = np.exp(2j*PI*PRF/4*t_ar)

fig, ax = plt.subplots(1,2)
fig.suptitle("test datacube processing")
ax[0].set_title("unprocessed datacube")
ax[0].imshow(abs(dc), origin='lower')
ax[0].set_xlabel("slow time [PRI]")
ax[0].set_ylabel("fast time [fs]")

#process datacube in place
f_ax, r_ax =dopplerProcess_dataCube(dc, fs, PRF)

ax[1].set_title("processed datacube")
ax[1].pcolormesh(f_ax*1e-6, r_ax, abs(dc))
ax[1].set_xlabel("frequency [MHz]")
ax[1].set_ylabel("range [m]")
plt.tight_layout()

plt.show()
