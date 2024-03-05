#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from scipy import fft
from scipy import signal
import matplotlib.pyplot as plt

from pulse_doppler_radar import frequency_aliased, range_unambiguous, frequency_doppler
from rf_datacube import calc_range_axis, create_dataCube, dopplerProcess_dataCube, R_pf_tgt
from waveforms import makeUncodedPulse
from waveform_helpers import insertWvfAtIndex, matchFilterPulse
from range_equation import snr_rangeEquation_CP

# constants
C = 3e8
PI = np.pi

tgtInfo = {"range": 60, "rangeRate" : 8.67e4, "rcs" : 10}
radar = {"fcar" : 1e9,
         "txPower" : 1,
         "txGain" : 1,
         "rxGain" :1,
         "opTemp" : 1,
         "sampRate" : 1000e6,
         "noiseFig": 3,
         "totalLosses" : 8,
         "PRF" : 1e6}

wvf = {"type" : "uncoded",
       "bw" : 40e6}
Npulses = 32


# def signal_gen(tgtInfo : dict, radar : dict, wvf : dict, Npulses : int):
#     """
#     Inputs:\n
#     \ttgtInfo  : dict holding range, rangeRate, and rcs (range rate constant for CPI)
#     \tradar  : dict holding fcar, txPower, txGain, rxGain, opTemp, sampRate, noiseFig, totalLosses, PRF
#     \twvf : string for wvform types
#     \tNpulses : number of puleses the CPI
#     \n
#     Outputs:\n
#     \t?
#     """
t_ar = np.arange(Npulses)*1/radar["PRF"]

#1 Created grids
dc = create_dataCube(radar["sampRate"], radar["PRF"], Npulses)
r_axis = calc_range_axis(radar["sampRate"], dc.shape[0])

#2 Calcualte range and rang rate of the target
range_ar = tgtInfo["range"] + tgtInfo["rangeRate"]*t_ar

#3 calc first response pulse location
firstEchoBin = int(tgtInfo["range"]/range_unambiguous(radar["PRF"]))

#4 create normalized waveform
if wvf["type"] == "uncoded":
    _, pulse_wvf = makeUncodedPulse(radar['sampRate'], wvf['bw'])
    BW = wvf['bw']
    time_BW_product = 1
else:
    print("wvf type not found: using uncoded pulse")
    _, pulse_wvf = makeUncodedPulse(radar['sampRate'], wvf['bw'])
    BW = wvf['bw']
    time_BW_product = 1

#5 detrmin scaling factor for SNR
SNR = snr_rangeEquation_CP(radar["txPower"], radar["txGain"], radar["rxGain"], tgtInfo["rcs"],
                           C/radar["fcar"], tgtInfo["range"], BW, radar["noiseFig"],
                           radar["totalLosses"], radar["opTemp"], Npulses, time_BW_product)
SNR_volt = np.sqrt(SNR) #USE TODO

#6 place pulse in first column
aliasedRange_ar = range_ar%range_unambiguous(radar["PRF"])
phase_ar = -4*PI*radar["fcar"]/C*range_ar
for i in range(Npulses-firstEchoBin):
    # print(f"{i=}")
    rangeIndex = np.argmin(abs(r_axis - aliasedRange_ar[i])) # is it binned this way?
    pulse= pulse_wvf*np.exp(1j*phase_ar[i])
    dc[:,i+firstEchoBin] = insertWvfAtIndex(dc[:,i+firstEchoBin], pulse, rangeIndex)

dc_unproc = dc.copy()

#8 match filter
for j in range(dc.shape[1]):
    mf, _= matchFilterPulse(dc[:,j], pulse_wvf)
    dc[:, j] = mf

dc_mf = dc.copy()

#9 doppler process
#9.1 apply filter window?
#9.2 doppler process
#TODO do not need to return dc since it is replaced in function
dc, f_axis, r_axis = dopplerProcess_dataCube(dc, radar["sampRate"], radar["PRF"])


#f = -2* fc/c Rdot
#Rdot = -c*f/(2*fc)
rdot_axis = -C*f_axis/(2*radar["fcar"])*radar["PRF"]/radar["sampRate"] #TODO WHY ratio at end?

# return dc, f_axis, r_axis


plt.close('all')
plt.figure()
plt.pcolormesh(t_ar, r_axis, abs(dc_unproc))
plt.xlabel("dwell start time [s]")
plt.ylabel("range [m]")
plt.title("datacube")

plt.figure()
plt.pcolormesh(t_ar, r_axis, abs(dc_mf))
plt.xlabel("dwell start time [s]")
plt.ylabel("range [m]")
plt.title("mf datacube")

plt.figure()
plt.title("processed mf datacube")
# dc, f_axis, r_axis = signal_gen(tgtInfo, radar, wvf, Npulses)
plt.pcolormesh(rdot_axis, r_axis, abs(dc))
plt.xlabel("Rdot [m/s]")
plt.ylabel("range [m]")
