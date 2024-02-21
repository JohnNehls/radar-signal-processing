#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# constants
k_boltz = 1.38064852e-23 # m^2 kg / s^2 K
C = 3e8

def snr_rangeEquation(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T):
    """Single-pulse SNR"""
    return (Pt*Gt*Gr*sigma*wavelength**2)/(((4*np.pi)**3)*(R**4)*k_boltz*T*B*F*L)

def snr_rangeEquation_CP(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, Npulses, time_bandwidth_prod):
    """"Signal-to-noise ratio of range equation with coherent processing (CP)"""
    singlePulse_snr= snr_rangeEquation(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T)
    return singlePulse_snr*Npulses*time_bandwidth_prod

def snr_rangeEquation_BPSK_pulses(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, Npulses, n_c):
    """number of chips is the time-bandwidth product"""
    return snr_rangeEquation_CP(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, Npulses, n_c)

def snr_rangeEquation_CP_DF(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, Tcpi, tau_df):
    """"Signal-to-noise ratio of range equation with coherent processing (CP) in the
    duty factor form."""
    singlePulse_snr= snr_rangeEquation(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T)
    return singlePulse_snr*Tcpi*tau_df

def lin2dB(linear):
    return 10*np.log10(linear)


## problem 1 #############################################################
antenna_diameter= 1.5*(0.0254*12) # convert feet to meters
fc = 10e9
B = 10e6
Ncode = 13
L = 10**(8/10) # 8dB
F = 10**(6/10) # 5dB
Npulses = 256
Pt_ar = [1e3, 5e3, 10e3]
sig_db_ar = [0, 10, 20]
sig_ar = [10**(x/10) for x in sig_db_ar]
range_ar = np.linspace(1e3,30e3,100)
threshold_db = 12
threshold = 10**(threshold_db/10)

# calculations
wavelength = C/fc
theta_3db = wavelength/antenna_diameter # for a circular apature radar
Gt = 4*np.pi/(theta_3db)**2 # az and el are the same
# assumptions
Gr = Gt # simpilifying assumption
T =290 # Kelvin

# plot
fig, ax = plt.subplots(1,len(Pt_ar), sharex='all', sharey='all')

for index, Pt in enumerate(Pt_ar):
    for sig in sig_ar:
        y = snr_rangeEquation_BPSK_pulses(Pt, Gt, Gr, sig, wavelength, range_ar, B, F, L, T, Npulses, Ncode)
        y = lin2dB(y)
        ax[index].plot(range_ar/1e3, y, label=f"{sig=}")
    ax[index].plot(range_ar/1e3, threshold_db*np.ones(range_ar.shape), label=f"threshold")
    ax[index].set_title(f"{Pt=:.1e}")
    ax[index].set_xlabel("target distance [km]")
    ax[index].set_ylabel("SNR dB")
    ax[index].grid()
    ax[index].legend()

## problem 2 #############################################################
