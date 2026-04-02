#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import rsp.uniform_linear_arrays as ula
import rsp.rdm as rdm
import rsp.monopulse as mp
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import uncoded_waveform, barker_coded_waveform, lfm_waveform
from rsp.returns import Target, Return



################################################################################
# example that monopulse can be done after doppler processing
################################################################################
# - Still need to investigate how this phase propagates from time domain to doppler
# - Is the example considering operating frequency?
# TODO:
# - [ ]incorporate the normalized antenna gain
#   - Peak gain will be in radar dict

bw = 10e6

radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=50e3,
    dwell_time=2e-3,
)

waveform = uncoded_waveform(bw)                        # high 1
waveform = barker_coded_waveform(bw, nchips=13)              # high 1
waveform = lfm_waveform(bw, T=10 / 40e6, chirp_up_down=1)  # high 2

tgt_angle = 5
dx = 1 / 2  # seperation of array elements in terms of wavelength
array_pos = np.array([-dx / 2, dx / 2])  # in terms of wavelength
steer_vec = ula.steering_vector(array_pos, tgt_angle)

dc_list = []

for sv in steer_vec:
    rseed = np.random.randint(1000)
    return_list = [Return(target=Target(range=2.4e3, range_rate=0.2e3, rcs=10, sv=sv))]
    rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(
        radar, waveform, return_list, snr=True, debug=False, plot=False, seed=rseed
    )
    dc_list.append(total_dc)

for i in range(len(steer_vec)):
    rdm.plot_rdm(rdot_axis, r_axis, dc_list[i], f"{i=}", cbar_min=-100, volt_to_dbm=True)

f_measured_theta = mp.monopulse_angle_at_peak_deg(dc_list[0], dc_list[1], dx)
f_measured_error = abs(f_measured_theta - tgt_angle)

print(f"{f_measured_theta=} degrees")
print(f"{f_measured_error=} degrees")

plt.show()
