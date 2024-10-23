#!/usr/bin/env python

import sys
from scipy import signal
import matplotlib.pyplot as plt
import rsp.uniform_linear_arrays as ula

# Discrpencies with Document
# - our figure 24 and 25 recreations do not line up with the document
#   - our figure 27 and 28 does
# - documents the 40 element, lamda/2 spacing in figure 24 and unwighted in 26 DISAGREE!
# - I am assuming document introduced an error in figures 24 and 25
#   - 24 and 25 use 10 log10(arrayFactor), not 20xlog10(arrayFactor)


# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True


plt.rcParams["text.usetex"] = True

## problem 1 ########################################################
## recreate figure 24  ######
fig, axs = plt.subplots(1,2)

fig.suptitle(r"Unweighted Array Factor for $\lambda/2$ spacing")
theta, gain = ula.linear_antenna_gain(10, 1/2, plot=False)
axs[0].plot(theta, gain)
axs[0].set_title('10 elements')

theta, gain = ula.linear_antenna_gain(40, 1/2, plot=False)
axs[1].plot(theta, gain)
axs[1].set_title('40 elements')

for ax in axs:
    ax.grid()
    ax.set_ylim((-60, 40))
    ax.set_xlabel(r'Angle $\theta$ [deg]')
    ax.set_ylabel("Gain [dBi]")

plt.tight_layout()

## recreate figure 25  ######
fig, axs = plt.subplots(1,3)

fig.suptitle("Unweighted Array Factor for 10 Elements")
theta, gain = ula.linear_antenna_gain(10, 1/4, plot=False)
axs[0].plot(theta, gain - gain.max())
axs[0].set_title(r'$\lambda/4$ spacing')

theta, gain = ula.linear_antenna_gain(10, 1/2, plot=False)
axs[1].plot(theta, gain - gain.max())
axs[1].set_title(r'$\lambda/2$ spacing')

theta, gain = ula.linear_antenna_gain(10, 1, plot=False)
axs[2].plot(theta, gain - gain.max())
axs[2].set_title(r'$\lambda$ spacing')

for ax in axs:
    ax.grid()
    ax.set_ylim((-80, 10))
    ax.set_xlabel(r'Angle $\theta$ [deg]')
    ax.set_ylabel("Normalized Gain [dBi]")

plt.tight_layout()

## recreate figure 27  ######
Nel = 40
chebWindow = signal.windows.chebwin(Nel, 30)  # less than 45dB and function reports warning
tayWindow = signal.windows.taylor(Nel, sll=35)

fig, axs = plt.subplots(1,2)
fig.suptitle(r"Array Factor with Different Weights: 40 elements, $\lambda/2$ spacing")

theta, gain = ula.linear_antenna_gain(Nel, 1/2, plot=False)
theta, gain_cheb = ula.linear_antenna_gain(Nel, 1/2, weights=chebWindow, plot=False)
theta, gain_tay = ula.linear_antenna_gain(Nel, 1/2, weights=tayWindow, plot=False)
axs[0].set_xlim((-90,90))
axs[1].set_xlim((-8,8))

for ax in axs:
    ax.plot(theta, gain - gain.max(), label="unweighted")
    ax.plot(theta, gain_cheb - gain_cheb.max(), '-.r', label="30 dB Cheb.")
    ax.plot(theta, gain_tay - gain_tay.max(), '--k', label="35 dB Taylor")
    ax.set_xlabel(r'Angle $\theta$ [deg]')
    ax.set_ylabel("Normalized Gain [dBi]")
    ax.set_ylim((-60, 5))
    ax.grid()

axs[1].legend()
plt.tight_layout()

## recreate figure 28  ######
fig, axs = plt.subplots(1,3)

fig.suptitle(r"Weighted Array Factor: $\lambda/2$ spacing, 20 elements")
Nel = 20
dx = 1/2
theta, gain = ula.linear_antenna_gain(Nel, dx, steer_angle=15, plot=False)
axs[0].plot(theta, gain - gain.max())
axs[0].set_title(r'Steered to 15 deg')

theta, gain = ula.linear_antenna_gain(Nel, dx, steer_angle=45, plot=False)
axs[1].plot(theta, gain - gain.max())
axs[1].set_title(r'Steered to 45 deg')

theta, gain = ula.linear_antenna_gain(Nel, dx, steer_angle=-60, plot=False)
axs[2].plot(theta, gain - gain.max())
axs[2].set_title(r'Steered to -60 deg')

for ax in axs:
    ax.grid()
    ax.set_ylim((-60,5))
    ax.set_xlabel(r'Angle $\theta$ [deg]')
    ax.set_ylabel("Normalized Gain [dBi]")

plt.tight_layout()

plt.show(block=BLOCK)

