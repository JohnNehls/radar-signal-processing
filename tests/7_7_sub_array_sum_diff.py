#!/usr/bin/env python

import sys
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import rsp.uniform_linear_arrays as ula
import rsp.utilities as rspu
import rsp.constants as c
# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

plt.rcParams["text.usetex"] = True

# Notes
# - another error in the equation in the document
#   - missing the normalization of the weights
fc = 10*10**9
wavelength = c.C/fc
Nel = 20

dx = wavelength/2  # [m]
L = (Nel - 1) * dx
el_pos = np.linspace(-L / 2, L / 2, Nel)  # wavelengths

assert Nel % 2 == 0
sub_arrays = [ el_pos[:int(Nel/2)], el_pos[int(Nel/2):] ]
weight = np.ones(int(Nel/2))

sub_array_gain = []
for sub_array in sub_arrays:
    # sub_array_phase_center.append(ula.array_phase_center(sub_array,
    thetas, gain = ula.linear_antenna_gain_meters(sub_array, fc)
    sub_array_gain.append(gain)

sum_gain = sub_array_gain[0] + sub_array_gain[1]
diff_gain = sub_array_gain[0] - sub_array_gain[1]


plt.plot(thetas, abs(sum_gain), label=r"$\Sigma$")
plt.plot(thetas, abs(diff_gain), label=r"$\Delta$")
plt.xlabel(r"Angle $\theta$ [deg]")
plt.ylabel("Gain")
plt.xlim((-20,20))
plt.grid()
plt.legend()
plt.tight_layout()

plt.show(block=BLOCK)
