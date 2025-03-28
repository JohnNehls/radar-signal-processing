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

plt.rcParams["text.usetex"] = True

# Notes
# - Estimation accuracy greatly depends on the target angle
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

    recieved_signals = []  # noisey signals with specified snr
    for sv in steer_vec:
        recieved_signals.append(snr_volt_scale * sv * signal_ar +
                               unity_variance_complex_noise(N_samples))
    plt.plot(recieved_signals[0], label=f"snr={snr_db}")

    dx = array_pos[1] - array_pos[0]  # seperation of array elements
    rho = 2 * np.pi * dx
    sum = recieved_signals[0] + recieved_signals[1]
    delta = recieved_signals[0] - recieved_signals[1]
    v_theta = np.arctan(2 * (delta / sum).imag) / (rho)  # ALGEBRA ERROR IN DOC
    measured_theta = np.arcsin(v_theta)

    measured_error = abs(np.rad2deg(measured_theta) - tgt_angle)
    error_mean_list.append(np.mean(measured_error))
    error_std_list.append(np.std(measured_error))

# signals per db
plt.legend()
plt.title("Noisy signal for Each SNR [dB]")
plt.xlabel("time [s]")
plt.ylabel("amplitude [v]")
plt.grid()

# plot the results
fig, axs = plt.subplots(1, 2)
fig.suptitle("Monopulse Angle Estimation")
axs[0].plot(snr_db_list, error_mean_list)
axs[0].set_title("Mean Angle Error")
axs[1].plot(snr_db_list, error_std_list)
axs[1].set_title(r"$\sigma$ Angle Error")

for ax in axs:
    ax.grid()
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Angle [Deg]")

plt.tight_layout()
plt.show(block=BLOCK)
