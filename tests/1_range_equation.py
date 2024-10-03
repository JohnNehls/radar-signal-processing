#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from rsp.constants import PI, C
import rsp.range_equation as re

## problem 1 #############################################################
# given
antenna_diameter= 1.5*(0.0254*12) # convert feet to meters
fc = 10e9
B = 10e6
Ncode = 13
L = 10**(8/10) # 8dB
F = 10**(6/10) # 5dB
n_p = 256
Pt_ar = [1e3, 5e3, 10e3]
sig_db_ar = [0, 10, 20]
sig_ar = [10**(x/10) for x in sig_db_ar]
R_ar = np.arange(1e3,30.1e3,100)
SNR_thresh_db = 12
SNR_thresh = 10**(SNR_thresh_db/10)

# calculations
wavelength = C/fc
theta_3db = wavelength/antenna_diameter # for a circular apature radar
Gt = 4*PI/(theta_3db)**2 # az and el are the same
# assumptions
Gr = Gt # simpilifying assumption
T =290 # Kelvin

# plot
fig, ax = plt.subplots(1,len(Pt_ar), sharex='all', sharey='all')
fig.suptitle('BPSK SNR')
for index, Pt in enumerate(Pt_ar):
    for sig_index, sig in enumerate(sig_ar):
        y = re.snr_rangeEquation_BPSK_pulses(Pt, Gt, Gr, sig, wavelength, R_ar, B, F, L, T, n_p, Ncode)
        y = 10*np.log10(y) # convert to dB
        ax[index].plot(R_ar/1e3, y, label=f"RCS={sig_db_ar[sig_index]}[dBsm]")
    ax[index].plot(R_ar/1e3, SNR_thresh_db*np.ones(R_ar.shape),'--k', label=f"threshold")
    ax[index].set_title(f"Pt={Pt*1e-3:.1f}[kW]")
    ax[index].set_xlabel("target distance [km]")
    ax[index].set_ylabel("SNR dB")
    ax[index].grid()
    ax[index].legend(loc='upper right')

## problem 2 #############################################################
sigma = 10**(0/10) # 0 dBsm
Pt = 5e3 # Watts
Tcpi_ar = [2e-3, 5e-3, 10e-3] # seconds
dutyFactor_ar = [0.01, 0.1, 0.2] # 1%, 10%, 20%

#plot
fig, ax = plt.subplots(1,len(Tcpi_ar), sharex='all', sharey='all')
fig.suptitle('CPI DutyFactor SNR')
for index, Tcpi in enumerate(Tcpi_ar):
    for dutyFactor in dutyFactor_ar:
        y = re.snr_rangeEquation_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength, R_ar, F, L, T, Tcpi, dutyFactor)
        y = 10*np.log10(y) # convert to dB
        ax[index].plot(R_ar/1e3, y, label=f"DF={dutyFactor}")
    ax[index].plot(R_ar/1e3, SNR_thresh_db*np.ones(R_ar.shape), '--k', label=f"threshold")
    ax[index].set_title(f"CPI={Tcpi*1e3:.1f}[ms]")
    ax[index].set_xlabel("target distance [km]")
    ax[index].set_ylabel("SNR dB")
    ax[index].grid()
    ax[index].legend(loc='upper right')

## problem 3 #############################################################
SNR_thresh_db = 15
SNR_thresh = 10**(SNR_thresh_db/10)
dutyFactor = 0.1 # 10%
Tcpi = 2e-3 # seconds

# p3 figure 1
# y-axis transmit power, x-axis target RCS
Pt_ar = np.arange(500, 10.1e3,100)
sigma_db_ar = np.arange(-5,26,1)
sigma_ar = [10**(x/10) for x in sigma_db_ar]
# min_det_range_sigmaPt = np.zeros((len(sigma_ar),len(Pt_ar)))
min_det_range_sigmaPt = np.zeros((len(Pt_ar),len(sigma_ar)))

for i, Pt in enumerate(Pt_ar):
    for j, sigma in enumerate(sigma_ar):
        val = re.minTargetDetectionRange_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength,
                                                           SNR_thresh, F, L, T, Tcpi,
                                                           dutyFactor)
        min_det_range_sigmaPt[i,j] = val

plt.figure()
plt.title(f"Tcpi={Tcpi*1e3}[ms] DF={dutyFactor}  SNR_thresh={SNR_thresh_db}[dBsm]")
plt.pcolormesh(sigma_db_ar, Pt_ar*1e-3, min_det_range_sigmaPt*1e-3)
c = plt.colorbar()
c.set_label("minimum detectable target range [km]")
plt.xlabel("target RCS [dBsm]")
plt.ylabel("transmit power [kW]")


# p3 figure 2
# y-axis Tcpi, x-axis target RCS
Pt = 5e3 # Watts
Tcpi_ar = np.arange(1e-3, 50.2e-3,200e-6)
# min_det_range_sigmaTcpi = np.zeros((len(sigma_ar),len(Tcpi_ar)))
min_det_range_sigmaTcpi = np.zeros((len(Tcpi_ar),len(sigma_ar)))

for i, Tcpi in enumerate(Tcpi_ar):
    for j, sigma in enumerate(sigma_ar):

        val =  re.minTargetDetectionRange_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength,
                                                            SNR_thresh,F, L, T, Tcpi, dutyFactor)
        min_det_range_sigmaTcpi[i,j] = val

plt.figure()
plt.title(f"Pt={Pt*1e-3:.1f}kW  DF={dutyFactor}  SNR_thresh={SNR_thresh_db}[dBsm]")
plt.pcolormesh(sigma_db_ar, Tcpi_ar*1e3, min_det_range_sigmaTcpi*1e-3)
c = plt.colorbar()
c.set_label("minimum detectable target range [km]")
plt.xlabel("target RCS [dBsm]")
plt.ylabel("CPI time [ms]")

plt.show()
