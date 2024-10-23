#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import rsp.uniform_linear_arrays as ula
from rsp.noise import unity_variance_complex_noise

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

# Notes
# - Is there a way to do this for modulated pulses?
#   - LFM pulses?

# signal
N_samples = 1000
time_ar = np.linspace(0, 10, N_samples)
freq = 1
signal_ar = np.exp(1j * 2 * np.pi * freq * time_ar)

# linear array
tgt_angle = 2
array_pos = np.array([-1 / 4, 1 / 4])  # in terms of wavelength
steer_vec = ula.steering_vector(array_pos, tgt_angle)

# calculated the error for SNR
snr_db_list = np.arange(30, step=3)

error_mean_list = []
error_std_list = []
for snr_db in snr_db_list:
    snr_volt_scale = 10 ** (snr_db / 20)

    recived_signals = []  # noisey signals with specified snr
    for sv in steer_vec:
        recived_signals.append(snr_volt_scale * sv + unity_variance_complex_noise(N_samples))

    rho = 2 * np.pi * (1 / 2)  # seperation of array elements
    sum = recived_signals[0] + recived_signals[1]
    delta = recived_signals[0] - recived_signals[1]
    v_theta = np.arctan(2 * (delta / sum).imag) / (rho)  # ALGEBRA ERROR IN DOC
    measured_theta = np.arcsin(v_theta)

    measured_error = abs(np.rad2deg(measured_theta) - tgt_angle)
    error_mean_list.append(np.mean(measured_error))
    error_std_list.append(np.std(measured_error))

# plot the results
fig, axs = plt.subplots(1, 2)
axs[0].plot(snr_db_list, error_mean_list)
axs[0].set_title("Mean Angle Error vs SNR")
axs[1].plot(snr_db_list, error_std_list)
axs[1].set_title("Angle Error Standard Deviation vs SNR")

for ax in axs:
    ax.grid()
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Angle [Deg]")

plt.tight_layout()
plt.show(block=BLOCK)
