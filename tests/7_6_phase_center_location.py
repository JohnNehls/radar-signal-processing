#!/usr/bin/env python

import sys
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import rsp.uniform_linear_arrays as ula

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

plt.rcParams["text.usetex"] = True

# Notes
# - another error in the equation in the document
#   - missing the normalization of the weights

Nel = 40
chebWindow = signal.windows.chebwin(Nel, 30)  # un-normalized
pos_ar = np.linspace(2, 8, Nel) * 1e-3

phase_cent = ula.array_phase_center(pos_ar, chebWindow)

plt.figure()
plt.plot(pos_ar, chebWindow, "o", label="weights")
plt.axvline(x=phase_cent, linestyle="dashed", color="k", label="phase center")
plt.xlabel("Array Element Position [m]")
plt.ylabel("Weight")
plt.legend()
plt.tight_layout()
plt.show(block=BLOCK)
