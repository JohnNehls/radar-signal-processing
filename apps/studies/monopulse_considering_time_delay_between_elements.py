#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from rsp import constants as c
import rsp.uniform_linear_arrays as ula
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import uncoded_waveform, barker_coded_waveform, lfm_waveform
from rsp.rdm import plot_rdm
import rsp.monopulse as mp
from rsp.rf_datacube import number_range_bins, range_axis, data_cube
from rsp.rf_datacube import matchfilter, doppler_process
from rsp.range_equation import noise_power
from rsp._rdm_internals import create_window, add_returns
from rsp.returns import Target, Return


################################################################################
# Show that the effect of considering the time delay
################################################################################
# - if fcar >= bw, time shift can be ignored
#   - The lambda/2 distance for the carrier then makes the element distance lambda/100 for IF freqs
#   - not much phase change
# - regardless of fcar, the monopulse estimated theta is not really effected

bw = 200e6

radar = Radar(
    fcar=1e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=50e3,
    dwell_time=0.5e-3,
)

return_list = [Return(target=Target(range=2.4e3, range_rate=0.2e3, rcs=10))]

waveform = uncoded_waveform(bw)  # high 1
waveform = barker_coded_waveform(bw, nchips=13)  # high 1
waveform = lfm_waveform(3 * bw, T=10 / 40e6, chirp_up_down=1)  ## high 2

tgt_angle = 2
dx = 1 / 2  # seperation of array elements in terms of carrier wavelength
array_pos = np.array([-dx / 2, dx / 2])  # in terms of wavelength

########## Compute waveform and radar parameters ###############################################
waveform.set_sample(radar.sample_rate)

########## Create range axis for plotting ######################################################
r_axis = range_axis(radar.sample_rate, number_range_bins(radar.sample_rate, radar.prf))

########## Return ##############################################################################
signal_dc = data_cube(radar.sample_rate, radar.prf, radar.n_pulses)

### Determin scaling factors for max voltage ###
rxVolt_noise = np.sqrt(c.RADAR_LOAD * noise_power(waveform.bw, radar.noise_factor, radar.op_temp))
noise_dc = np.random.uniform(low=-1, high=1, size=signal_dc.shape) * rxVolt_noise
add_returns(signal_dc, waveform, return_list, radar)

### Alter the signal due to linear array position #####################
assert abs(tgt_angle) < 90
signal_dc_ula_list = []
signal_dc_ula_list_timeshift = []
for pos in array_pos:
    # apply phase shift from carrier
    signal_dc_sv = signal_dc.copy() * ula.steering_vector(pos, tgt_angle)
    signal_dc_ula_list.append(signal_dc_sv)

    # apply time shift to baseband signal from array position from zero
    position_meters = c.C / radar.fcar * pos
    tmp_signal = signal_dc_sv.T.flatten()
    shifted_signal = ula.apply_timeshift_due_to_element_position(
        tmp_signal, radar.sample_rate, position_meters, tgt_angle
    )
    signal_dc_shift = shifted_signal.reshape(tuple(reversed(signal_dc_sv.shape))).T
    signal_dc_ula_list_timeshift.append(signal_dc_shift)

# plot imaginary since the real parts are identical for symetric linear arrays (Cos)
fun = np.imag
plt.figure()
plt.title("Imaginary component of signal")
plt.plot(fun(signal_dc_ula_list[0].T.flatten()), label="neg position")
plt.plot(fun(signal_dc_ula_list[1].T.flatten()), label="pos position")
plt.plot(fun(signal_dc_ula_list_timeshift[0].T.flatten()), "--", label="timeshift neg position")
plt.plot(fun(signal_dc_ula_list_timeshift[1].T.flatten()), "--", label="timeshift pos position")
plt.xlabel("sample")
plt.ylabel("amplitude")
plt.legend()

# list of datacubes to process
rdm_list = signal_dc_ula_list + signal_dc_ula_list_timeshift

########## Apply the match filter ##############################################################
for dc in rdm_list:
    matchfilter(dc, waveform.pulse_sample, pedantic=True)

########### Doppler process ####################################################################
# First create filter window and apply it
chwin_norm_mat = create_window(signal_dc.shape, plot=False)
for dc in rdm_list:
    dc *= chwin_norm_mat

# Doppler process datacubes
for dc in rdm_list:
    f_axis, r_axis = doppler_process(dc, radar.sample_rate)

########## Plots and checks ####################################################################
# calc rangeRate axis  #f = -2* fc/c Rdot -> Rdot = -c+f/ (2+fc)
rdot_axis = -c.C * f_axis / (2 * radar.fcar)


########## Monopulse on the RDMs #######################################################
def monopulse(dx, dc_list, tgt_angle):
    f_measured_theta = mp.monopulse_angle_at_peak_deg(dc_list[0], dc_list[1], dx)
    f_measured_error = abs(f_measured_theta - tgt_angle)
    print(f"\t{f_measured_theta=} degrees")
    print(f"\t{f_measured_error=} degrees")


print("No-timeshift")
monopulse(dx, signal_dc_ula_list, tgt_angle)
print("timeshift")
monopulse(dx, signal_dc_ula_list_timeshift, tgt_angle)

plot_rdm(rdot_axis, r_axis, signal_dc_ula_list[0], "Noiseless RDM")

plt.show()
