#!/usr/bin/env python
import enum
import numpy as np
import matplotlib.pyplot as plt

# constants
k_boltz = 1.38064852e-23 # m^2 kg / s^2 K
C = 3e8

def snr_rangeEquation(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T):
    """Single-pulse SNR"""
    return (Pt*Gt*Gr*sigma*wavelength**2)/(((4*np.pi)**3)*(R**4)*k_boltz*T*B*F*L)

def snr_rangeEquation_CP(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p,
                         time_bandwidth_prod):
    """"Signal-to-noise ratio of range equation with coherent processing (CP)"""
    singlePulse_snr= snr_rangeEquation(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T)
    return singlePulse_snr*n_p*time_bandwidth_prod

def snr_rangeEquation_BPSK_pulses(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, n_c):
    """number of chips is the time-bandwidth product"""
    return snr_rangeEquation_CP(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, n_c)

def snr_rangeEquation_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength, R, F, L, T,
                                        Tcpi, tau_df):
    """"Signal-to-noise ratio of range equation with coherent processing (CP) in the
    duty factor form."""
    singlePulse_snr= snr_rangeEquation(Pt, Gt, Gr, sigma, wavelength, R, 1, F, L, T)
    return singlePulse_snr*Tcpi*tau_df

def minTargetDetectionRange(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T):
    """single pulse"""
    return ((Pt*Gt*Gr*sigma*wavelength**2)/(((4*np.pi)**3)*(SNR_thresh)*k_boltz*T*B*F*L))**(1/4)

def minTargetDetectionRange_BPSK_pulses(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F,
                                        L, T, n_p, n_c):
    """BPSK pulses"""
    onePulse = minTargetDetectionRange(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T)
    return onePulse*(n_p*n_c)**(1/4)

def minTargetDetectionRange_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength, SNR_thresh,
                                              F, L, T, Tcpi, tau_df):
    """CPI time and dutyfactor"""
    onePulse = minTargetDetectionRange(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, 1, F, L, T)
    return onePulse*(Tcpi*tau_df)**(1/4)

def lin2dB(linear):
    return 10*np.log10(linear)


## problem 1 #############################################################
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
Gt = 4*np.pi/(theta_3db)**2 # az and el are the same
# assumptions
Gr = Gt # simpilifying assumption
T =290 # Kelvin

# plot
fig, ax = plt.subplots(1,len(Pt_ar), sharex='all', sharey='all')
fig.suptitle('BPSK SNR')
for index, Pt in enumerate(Pt_ar):
    for sig in sig_ar:
        y = snr_rangeEquation_BPSK_pulses(Pt, Gt, Gr, sig, wavelength, R_ar, B, F, L, T, n_p, Ncode)
        y = lin2dB(y)
        ax[index].plot(R_ar/1e3, y, label=f"{sig=}")
    ax[index].plot(R_ar/1e3, SNR_thresh_db*np.ones(R_ar.shape), label=f"threshold")
    ax[index].set_title(f"{Pt=:.1e}")
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
        y = snr_rangeEquation_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength, R_ar, F, L, T, Tcpi, dutyFactor)
        y = lin2dB(y)
        ax[index].plot(R_ar/1e3, y, label=f"df={dutyFactor}")
    ax[index].plot(R_ar/1e3, SNR_thresh_db*np.ones(R_ar.shape), label=f"threshold")
    ax[index].set_title(f"{Tcpi=:.1e}")
    ax[index].set_xlabel("target distance [km]")
    ax[index].set_ylabel("SNR dB")
    ax[index].grid()
    ax[index].legend(loc='upper right')

## problem 3 #############################################################
plt.close('all')
SNR_thresh_db = 15
SNR_thresh = 10**(SNR_thresh_db/10)
dutyFactor = 0.1 # 10%
Tcpi = 2e-3 # seconds


# p3 figure 1
# y-axis transmit power, x-axis target RCS
Pt_ar = np.arange(500, 10.1e3,100)
sigma_db_ar = np.arange(-5,26,1)
sigma_ar = [10**(x/10) for x in sigma_db_ar]
min_det_range_sigmaPt = np.zeros((len(sigma_ar),len(Pt_ar)))

for i, sigma in enumerate(sigma_ar):
    for j, Pt in enumerate(Pt_ar):
        min_det_range_sigmaPt[i,j] = minTargetDetectionRange_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength,
                                                                SNR_thresh, F, L, T, Tcpi,
                                                                dutyFactor)
def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

plt.figure()
plt.title(f"Tcpi={Tcpi*1e3}[ms] df={dutyFactor}  SNR_thresh={SNR_thresh_db}[dBsm]")
plt.imshow(min_det_range_sigmaPt*1e-3, aspect='auto', interpolation='none',origin='lower',
           extent=extents(sigma_db_ar) + extents(Pt_ar*1e-3),cmap='viridis')
c = plt.colorbar()
c.set_label("minimum detectable target range [km]")
plt.xlabel("target RCS [dBsm]")
plt.ylabel("transmit power [kW]")


# p3 figure 2
# y-axis Tcpi, x-axis target RCS
Pt = 5e3 # Watts
Tcpi_ar = np.arange(1e-3, 5.2e-3,200e-6)
min_det_range_sigmaTcpi = np.zeros((len(sigma_ar),len(Tcpi_ar)))

for i, sigma in enumerate(sigma_ar):
    for j, Tcpi in enumerate(Tcpi_ar):
        val =  minTargetDetectionRange_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength, SNR_thresh,
                                                         F, L, T, Tcpi, dutyFactor)
        min_det_range_sigmaTcpi[i,j] = val

plt.figure()
plt.title(f"Pt={Pt*1e-3:.1f}kW  df={dutyFactor}  SNR_thresh={SNR_thresh_db}[dBsm]")
plt.imshow(min_det_range_sigmaTcpi*1e-3, aspect='auto', interpolation='none',origin='lower',
           extent=extents(sigma_db_ar) + extents(Tcpi_ar*1e3),cmap='viridis')
c = plt.colorbar()
c.set_label("minimum detectable target range [km]")
plt.xlabel("target RCS [dBsm]")
plt.ylabel("CPI time [ms]")
