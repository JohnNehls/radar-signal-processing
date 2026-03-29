#!/usr/bin/env python

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import rsp.uniform_linear_arrays as ula


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
plt.show()
