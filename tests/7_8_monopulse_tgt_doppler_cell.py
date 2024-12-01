#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np
import rsp.uniform_linear_arrays as ula
from rsp import rdm
import rsp.rdm_helpers as rdmh

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True


################################################################################
# example that monopulse can be done after doppler processing
################################################################################
# - Still need to investigate how this phase propagates from time domain to doppler
# - Is the example considering operating frequency?
# TODO:
# - [ ]incorporate the normalized antenna gain
#   - Peak gain will be in radar dict

bw = 10e6

radar = {
    "fcar": 10e9,
    "txPower": 1e3,
    "txGain": 10 ** (30 / 10),
    "rxGain": 10 ** (30 / 10),
    "opTemp": 290,
    "sampRate": 2 * bw,
    "noiseFactor": 10 ** (8 / 10),
    "totalLosses": 10 ** (8 / 10),
    "PRF": 50e3,
    "dwell_time": 2e-3,
}

return_list = [{"type": "skin", "target": {"range": 2.4e3, "rangeRate": 0.2e3, "rcs": 10}}]

waveform = {"type": "uncoded", "bw": bw}  # high 1
waveform = {"type": "barker", "nchips": 13, "bw": bw}  # high 1
waveform = {"type": "lfm", "bw": bw, "T": 10 / 40e6, "chirpUpDown": 1} ## high 2

tgt_angle = 5
dx = 1/2  # seperation of array elements in terms of wavelength
array_pos = np.array([-dx/2, dx/2])  # in terms of wavelength
steer_vec = ula.steering_vector(array_pos, tgt_angle)

dc_list = [ ]

for sv in steer_vec:
    rseed = np.random.randint(1000)
    rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(
        radar, waveform, return_list, snr=True, debug=False, plot=False, steerVec=sv, seed=rseed
    )
    dc_list.append(total_dc)

for i in range(len(steer_vec)):
    rdmh.plot_rdm(rdot_axis, r_axis, dc_list[i], f"{i=}", cbarMin=-100, volt2dbm=True)

rho = 2 * np.pi * dx
sum = dc_list[0] + dc_list[1]
delta = dc_list[0] - dc_list[1]
v_theta = np.arctan(2 * (delta / sum).imag) / (rho)  # ALGEBRA ERROR IN DOC
f_max_index = np.where(abs(dc_list[0]) == abs(dc_list[0]).max())

f_measured_theta = np.rad2deg(np.arcsin(v_theta)[f_max_index])
f_measured_error = abs(f_measured_theta - tgt_angle)

print(f"{f_measured_theta=} degrees")
print(f"{f_measured_error=} degrees")

plt.show(block=BLOCK)
