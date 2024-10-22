#!/usr/bin/env python

import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import rsp.uniform_linear_arrays as ula

# Notes
# - Each array element has it's own channel
# - Making datacube data for each channel is done by adding phase from extra path length
#   - in phase, for planewave p = np.exp(1j* d sin(angle)/lambda )
#   - since our waveform may be mulit-chromatic, we use time rather than distance and wavelenthg

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True


#
plt.rcParams["text.usetex"] = True

## problem 1 ########################################################
## recreate figure 24  ######
fig, axs = plt.subplots(1,2)

fig.suptitle("Unweighted Array Factor")
theta, gain = ula.linear_antenna_gain(10, 1/2, plot=False)
axs[0].plot(theta, gain)
axs[0].set_title(r'$\lambda/2$ spaceing, 10 elements')

theta, gain = ula.linear_antenna_gain(40, 1/2, plot=False)
axs[1].plot(theta, gain)
axs[1].set_title(r'$\lambda/2$ spaceing, 40 elements')

for ax in axs:
    ax.grid()
    ax.set_ylim((-30,20))
    ax.set_xlabel(r'Angle $\theta$ [deg]')
    ax.set_ylabel("Gain [dBi]")

plt.tight_layout()
