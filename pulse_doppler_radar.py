#!/usr/bin/env python

# constants
K_BOLTZ = 1.38064852e-23 # m^2 kg / s^2 K
C = 3e8 # m/s
PI = 3.141592653589793

def range_unambiguous(PRF):
    "maximum unambigious range"
    return C/(2*PRF)

def frequency_doppler(rangeRate, f0):
    "frequnce of light recieved after reflection off target with the given rangeRate"
    return f0*(-2*rangeRate/C)

def frequency_aliased(freq, freq_sample):
    """Place freq in [-freq_sample/2, freq_sample/2]
    Usefull for finding aliasing of real signals"""
    f = freq%freq_sample
    if f > freq_sample/2:
        return  f - freq_sample
    else:
        return f

def rangeRate_pm_unambiguous(PRF, f0):
    "+/- bounds of the unambgeous velocity"
    return PRF*C/(4*f0)

def rangeRate_aliased_rrmax(rangeRate, rangeRate_max):
    """Place freq in [-rangeRate_max, rangeRate_max]
    Usefull for finding aliasing of real signals"""
    r = rangeRate%(2*rangeRate_max)
    if r > rangeRate_max:
        return  r - 2*rangeRate_max
    else:
        return r

def rangeRate_aliased_prf_f0(rangeRate, PRF, f0):
    """Place freq in [-rangeRate_max, rangeRate_max]
    Usefull for finding aliasing of real signals"""
    return rangeRate_aliased_rrmax(rangeRate, rangeRate_pm_unambiguous(PRF, f0))

#def main()
import numpy as np
import matplotlib.pyplot as plt

## problem 1 ######################################
PRF_ar = np.arange(1e3,200.5e3,500)
plt.figure()
plt.plot(PRF_ar*1e-3, range_unambiguous(PRF_ar)*1e-3)
plt.title(f"unambiguous range vs PRF")
plt.xlabel("PRF [kHz]")
plt.ylabel("range [km]")
plt.yscale('log')
plt.grid(which="both")


## problem 2 ######################################
f0_ar = np.array([1,2,3,4,12,16,35,95])*1e9
plt.figure()
for f0 in f0_ar:
    plt.plot(PRF_ar*1e-3, rangeRate_pm_unambiguous(PRF_ar, f0)*1e-3, label=f"{f0=:.1e}")
plt.title("unambiguous +/- range rate vs PRF")
plt.xlabel("PRF [kHz]")
plt.ylabel("+/- max range rate [km/s]")
plt.yscale('log')
plt.legend()
plt.grid(which="both")

## problem 3 ######################################
#target is 15.5km away. plot apparent range for
R_tgt = 15.5*1e3 # m
PRF_ar = np.array([2,4,8,16,32,50,60,64,95,100,128,150,228])*1e3  # Hz

apparent_range_ar = []
range_max_ar = []

for PRF in PRF_ar:
    range_max = range_unambiguous(PRF)
    apparent_range_ar.append(R_tgt%range_max)
    range_max_ar.append(range_max)

apparent_range_ar = np.array(apparent_range_ar)
range_max_ar = np.array(range_max_ar)

fig, ax = plt.subplots(1,2)
fig.suptitle(f"aliased range vs PRF: R_tgt = {R_tgt*1e-3:.1f} [km]")
ax[0].set_title("aliased range")
ax[0].plot(PRF_ar*1e-3, apparent_range_ar*1e-3,'o')
ax[0].set_xlabel("PRF [kHz]")
ax[0].set_ylabel("apparent range [km]")
ax[0].grid()
ax[1].set_title(f"unambiguous range [km]")
ax[1].plot(PRF_ar*1e-3, range_max_ar*1e-3)
ax[1].set_xlabel("PRF [kHz]")
ax[1].set_ylabel("unambiguous range [km]")
ax[1].grid()

## problem 4 ######################################
rangeRate_tgt = -750 # m/s
f0 = 10e9 # Hz
apparent_doppler_ar = []
apparent_rangeRate_ar = []

for PRF in PRF_ar:
    doppler_freq_tgt = frequency_doppler(rangeRate_tgt, f0)
    apparent_doppler_ar.append(frequency_aliased(doppler_freq_tgt, PRF))
    apparent_rangeRate_ar.append(rangeRate_aliased_prf_f0(rangeRate_tgt, PRF, f0))

apparent_doppler_ar = np.array(apparent_doppler_ar)
apparent_rangeRate_ar = np.array(apparent_rangeRate_ar)

fig, ax =plt.subplots(1,2)
fig.suptitle(f"Rdot_tgt = {rangeRate_tgt} [m/s]")
ax[0].set_title(f"aliased Doppler freq")
ax[0].plot(PRF_ar*1e-3, apparent_doppler_ar*1e-3, '-o')
ax[0].set_xlabel("PRF [kHz]")
ax[0].set_ylabel("apparent Doppler freq [kHz]")
ax[0].grid()
ax[1].set_title(f"aliased range rate")
ax[1].plot(PRF_ar*1e-3, apparent_rangeRate_ar, "-o",label="apparentRdot")
ax[1].plot(PRF_ar*1e-3, rangeRate_pm_unambiguous(PRF_ar, f0), "--r", label="unamb_Rdot")
ax[1].plot(PRF_ar*1e-3, -rangeRate_pm_unambiguous(PRF_ar, f0), "--r")
ax[1].set_xlabel("PRF [kHz]")
ax[1].set_ylabel("apparent rangeRate [m/s]")
ax[1].legend()
ax[1].grid()

## problem 5 ######################################
rangeRate_tgt = -750 # m/s
PRF = 16e3  # Hz
f0_ar = np.array([1,2,4,6,8,10,12,16,18,34,36,94])*1e9

apparent_rangeRate_ar = []
rangeRate_max_ar = []
for f0 in f0_ar:
    rangeRate_max = rangeRate_pm_unambiguous(PRF, f0)
    rangeRate_max_ar.append(rangeRate_max)
    apparent_rangeRate_ar.append(rangeRate_aliased_prf_f0(rangeRate_tgt, PRF, f0))

apparent_rangeRate_ar = np.array(apparent_rangeRate_ar)
rangeRate_max_ar = np.array(rangeRate_max_ar)

plt.figure()
plt.title(f"aliased range rate vs freq: Rdot_tgt = {rangeRate_tgt} [m/s] PRF={PRF*1e-3:.1f}[km]")
plt.plot(f0_ar*1e-9, apparent_rangeRate_ar,'o',label="apparentRdot")
plt.plot(f0_ar*1e-9, rangeRate_max_ar, '--r', label="unamb_Rdot")
plt.plot(f0_ar*1e-9, -rangeRate_max_ar, '--r')
plt.xlabel("operation frequency [GHz]")
plt.ylabel("apparent rangeRate [m/s]")
plt.legend()
plt.grid()
