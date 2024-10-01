#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from rsp.constants import PI, C, K_BOLTZ
import rsp.utilities as utl

## test phase branch aliases ################################
phase_ar = np.linspace(-3*PI, 5*PI, 100)
phase_ar_negpi_pospi = utl.phase_negpi_pospi(phase_ar)
plt.figure()
plt.title("[-pi,pi) phase branch test")
plt.plot(phase_ar, phase_ar_negpi_pospi, 'o')
plt.xlabel("input phase")
plt.ylabel("output phase")
plt.grid()
print("check the that -pi is included and pi is not" )

phase_ar = np.linspace(-4*PI, 4*PI, 100)
phase_ar_zero_twopi = utl.phase_zero_twopi(phase_ar)
plt.figure()
plt.title("[0, 2pi) phase branch test")
plt.plot(phase_ar, phase_ar_zero_twopi, 'o')
plt.xlabel("input phase")
plt.ylabel("output phase")
plt.grid()
print("check the that 0 is included and 2pi is not" )

plt.show()
