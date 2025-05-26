#!/usr/bin/env python

import sys
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


plt.rcParams["text.usetex"] = True

## STUDY 1 : show that as long as you have dx < lambda/2, the array factor structure is defined
# - it onlys scales with more elements
plt.figure()
plt.title(r"Unweighted Array Factor with Constant Length, L = 5 $\lambda$")
theta2, gain2 = ula.linear_antenna_gain_N_db(10, 1 / 2, plot=False)
theta4, gain4 = ula.linear_antenna_gain_N_db(20, 1 / 4, plot=False)
theta8, gain8 = ula.linear_antenna_gain_N_db(40, 1 / 8, plot=False)
plt.plot(theta2, gain2, "-b", label=r"dx = $\lambda/2$, 10 elements")
plt.plot(theta4, gain4, "-.r", label=r"dx = $\lambda/4$, 20 elements")
plt.plot(theta8, gain8, "--k", label=r"dx = $\lambda/8$, 40 elements")
plt.xlabel(r"Angle $\theta$ [deg]")
plt.ylabel("Gain [dBi]")
plt.ylim((-60, 40))
plt.legend()
plt.grid()
plt.tight_layout()


## STUDY 2 : show that the longer the array, the more defined the bore-sight peak
# - much like a regular Fourier Transform pair
plt.figure()
plt.title(r"Unweighted Array Factor with Constant Element Spacing, dx = $\lambda/2$")
theta2, gain2 = ula.linear_antenna_gain_N_db(4, 1 / 2, plot=False)
theta4, gain4 = ula.linear_antenna_gain_N_db(8, 1 / 2, plot=False)
theta8, gain8 = ula.linear_antenna_gain_N_db(16, 1 / 2, plot=False)
plt.plot(theta2, gain2, "-b", label=r"L = $2\lambda$, 4 elements")
plt.plot(theta4, gain4, "-.r", label=r"L = $4\lambda$, 8 elements")
plt.plot(theta8, gain8, "--k", label=r"L = $8\lambda$, 16 elements")
plt.xlabel(r"Angle $\theta$ [deg]")
plt.ylabel("Gain [dBi]")
plt.ylim((-60, 30))
plt.legend()
plt.grid()
plt.tight_layout()


plt.show(block=BLOCK)
