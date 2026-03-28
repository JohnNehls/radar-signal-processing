#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft


# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

# Start experimenting
Nel = 40
dx = 1 / 2
M = 100

L = (Nel - 1) * dx

el_pos = np.linspace(-L / 2, L / 2, Nel)  # wavelengths

weights = np.ones(Nel)
# weights = signal.windows.chebwin(Nel,60)

phase = np.exp(1j * 2 * np.pi * (Nel - 1) / 2 * dx)
af = fft.ifftshift(fft.ifft(weights))
afp = phase * af
vtheta_max = 1 / (2 * dx)

faxis = np.linspace(-vtheta_max, vtheta_max, Nel)

plt.plot(faxis, abs(af), "-o")

plt.show(block=BLOCK)

raise Exception("This funcionality/test is incomplete")
