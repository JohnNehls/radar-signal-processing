import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pulse_doppler_radar import range_unambiguous
from constants import PI, C
from waveform_helpers import addWvfAtIndex
from vbm import create_VBM_slowtime_noise
from utilities import phase_negpi_pospi

def firstEchoBin(range, PRF):
    return int(range/range_unambiguous(PRF))

def plotRTM (r_axis, data, title):
    """Plot range-time matrix"""
    pulses = range(data.shape[1])
    fig, ax = plt.subplots(1,2)
    fig.suptitle(title)
    p = ax[0].pcolormesh(pulses, r_axis*1e-3, abs(data))
    ax[0].set_xlabel("pulse number")
    ax[0].set_ylabel("range [km]")
    ax[0].set_title("magnitude")
    fig.colorbar (p)
    ax[1].pcolormesh(pulses, r_axis*1e-3, np.angle(data))
    ax[1].set_xlabel("pulse number")
    ax[1].set_ylabel("range [km]")
    ax[1].set_title("phase")
    fig.tight_layout ()


def setZeroToSmallestNumber (array):
    smallest_float32= sys.float_info.min + sys.float_info.epsilon
    indxs = np.where(array==0)
    array[indxs] = smallest_float32


def plotRDM(rdot_axis, r_axis, data, title, cbarRange=30, volt2db=True):
    """Plot range-Doppler matrix"""
    data = abs(data)
    fig, ax = plt.subplots(1,1)
    fig.suptitle(title)
    if volt2db:
        setZeroToSmallestNumber(data)
        data = 20*np.log10(data)
    p = ax.pcolormesh (rdot_axis*1e-3, r_axis*1e-3, data)
    ax.set_xlabel("range rate [km/s]")
    ax.set_ylabel("range [km]")
    ax.set_title("magnitude squared")
    if cbarRange:
        p.set_clim((data.max() - cbarRange, data.max()))
    cbar = fig.colorbar(p)
    if volt2db:
        cbar.set_label("SNR [dB]")
    else:
        cbar.set_label("SNR")
    fig.tight_layout ()


def addSkin_old(signal_dc, wvf:dict, tgtInfo:dict, radar:dict, tgt_range_ar, r_axis, SNR_volt):
    """asdfadsf"""

    firstEchoIndex = firstEchoBin(tgt_range_ar[0], radar["PRF"])

    ## pulses timed from their start not their center, we compensate with pw/2 range offset
    range_pw_offset = wvf["pulse_width"]/2*C/2
    aliasedRange_ar = tgt_range_ar%range_unambiguous(radar["PRF"])
    phase_ar = -4*PI*radar["fcar"]/C*tgt_range_ar

    for i in range(radar["Npulses"] - firstEchoIndex):
        # TODO is this how these should be binned? Should they be interpolated onto grid?
        rangeIndex = np.argmin(abs(r_axis - aliasedRange_ar[i] + range_pw_offset))

        pulse= SNR_volt*wvf["pulse"]*np.exp(1j*phase_ar[i])

        addWvfAtIndex(signal_dc[:,i+firstEchoIndex], pulse, rangeIndex) # add in place

def addSkin(signal_dc, wvf:dict, tgtInfo:dict, radar:dict, SNR_volt):
    """asdfadsf"""

    # time and range arrays
    time_ar = np.arange(signal_dc.size)*1/radar["sampRate"]  # time of all samples in CPI
    t_slow_axis = np.arange(radar["Npulses"])*1/radar["PRF"] # time when pulses sent

    tgt_range_ar = tgtInfo["range"] + tgtInfo["rangeRate"]*t_slow_axis # tgt range at pulse send
    twoWay_time_delay_ar = 2*tgt_range_ar/C # time of travel from radar to tgt and back
    pulse_return_time = t_slow_axis + twoWay_time_delay_ar # time pulses return to radar
    twoWay_phase_ar = -2*PI*radar["fcar"]*twoWay_time_delay_ar # Phase added due to

    ## pulses timed from their start not their center, we compensate with pw/2 range offset
    time_pw_offset = wvf["pulse_width"]/2

    # Due to the time axis being the non-continuous (slow) axis, we most do some transposing
    tmpSignal = signal_dc.T.flatten()

    for i in range(radar["Npulses"]):
        # TODO is this how these should be binned? Should they be interpolated onto grid?
        timeIndex = np.argmin(abs(time_ar - pulse_return_time[i] + time_pw_offset))
        pulse= SNR_volt*wvf["pulse"]*np.exp(1j*twoWay_phase_ar[i])

        if timeIndex < signal_dc.size: # else pulse is in next CPI
            addWvfAtIndex(tmpSignal, pulse, timeIndex)

    tmpSignal = tmpSignal.reshape(tuple(reversed(signal_dc.shape))).T

    signal_dc[:] = tmpSignal[:]

def addMemory(signal_dc, wvf:dict, tgtInfo:dict, radar:dict, returnInfo, r_axis, SNR_volt):
    """"Place pulse at range index and apply phase
         - this should be generalized to per-pulse phase and delay on first recorded waveform
        ## pulses timed from their start not their center, we compensate with pw/2 range offset
        - should EA compensate for this?
         - radar -> pod technique (time, freq) -> radar
         - change the amplitude to not be connected to SNR/rcs/target
         - ? x2 diff f_delta and f_rdot calc?
         interface for returnInfo: rdot_offset, rdot_delta, delay
    """
    t_slow_axis = np.arange(radar["Npulses"])*1/radar["PRF"]
    t_fast_axis = 2*r_axis/C
    firstEchoIndex = firstEchoBin(tgtInfo["range"], radar["PRF"])
    time_pw_offset = wvf["pulse_width"]/2
    oneWay_time_ar = (tgtInfo["range"] + tgtInfo["rangeRate"]*t_slow_axis)/C
    oneWay_phase_ar = -2*PI*radar["fcar"]*oneWay_time_ar

    #Make output offset from skin return #############################################
    if "rdot_offset" in returnInfo.keys():
        f_rdot = 2*radar["fcar"]/C*returnInfo["rdot_offset"] # remove x2 for absolute rdot
        rdot_offset_flag = True
    else:
        rdot_offset_flag = False

    #Achieve Velocity Bin Masking (VBM) by adding pahse in slow time #################
    # - want to add phase so wvfm will sill pass radar's match filter
    # - there are several methods see vbm.py
    if "rdot_delta" in returnInfo.keys():
        slowtime_noise = create_VBM_slowtime_noise(radar["Npulses"],
                                                   radar["fcar"],
                                                   returnInfo["rdot_delta"],
                                                   radar["PRF"],
                                                   debug=True)
    else:
        slowtime_noise = np.ones(radar["Npulses"]) # default if no VBM

    #Delay the return ################################################################
    # - can be negative
    # - can make a range interface which converts range to time
    if "delay" in returnInfo.keys():
        delay = returnInfo["delay"]
    else:
        delay = 0

    if "range_offset" in returnInfo.keys():
        delay = 2*returnInfo["range_offset"]/C
    else:
        delay = 0


    stored_pulse = 0; stored_angle = 0 # initialize to stop lsp from complaining

    for i in range(radar["Npulses"]-firstEchoIndex):
        # pulse recieved by the EW system
        recieved_pulse = wvf["pulse"]*np.exp(1j*oneWay_phase_ar[i])

        #Store first pulse and wait for next pulse
        if i == 0:
            stored_pulse = recieved_pulse
            continue

        #Calculate 1-way phase difference between first two pulses
        if i == 1:
            stored_angle = (np.angle(recieved_pulse) - np.angle(stored_pulse))
            stored_angle = phase_negpi_pospi(stored_angle)
            stored_angle = np.mean(stored_angle)

        # create base pulse
        # - TODO set amplitude base on pod parameters
        pulse = SNR_volt*stored_pulse

        # add noise (VBM)
        pulse = pulse*slowtime_noise[i]

        # add stored pulse difference rdot
        pulse = pulse*(np.exp(1j*i*stored_angle))

        # add prescirbed rdot offset
        if rdot_offset_flag:
            pulse = pulse*(np.exp(-1j*i*2*PI*f_rdot/radar["PRF"]))

        # add 1-way propagation phase back to radar
        # echo may be incorrect in line below
        pulse = pulse*(np.exp(1j*oneWay_phase_ar[i+firstEchoIndex])) #CLEAN UP echo piece!

        aliasedTime_ar = (2*oneWay_time_ar + delay)%(1/radar["PRF"])

        # TODO is this how these should be binned? Should they be interpolated onto grid?
        rangeIndex = np.argmin(abs(t_fast_axis - aliasedTime_ar[i] + time_pw_offset))

        addWvfAtIndex(signal_dc[:,i+firstEchoIndex], pulse, rangeIndex) # add in place

def noiseChecks(signal_dc, noise_dc, total_dc):
    print(f"\nnoise check:")
    noise_var = np.var (total_dc, 1)
    print(f"\t{np.mean (noise_var)=: .4f}")
    print(f"\t{np.var(noise_var)=: .4f}")
    print(f"\t{np.mean (20*np.log10(noise_var))=: .4f}")
    print(f"\t{np.var (20*np.log10(noise_var))=: .4f}")
    print(f"\nSNR test:")
    print(f"\t{20*np.log10(np.max(abs(signal_dc)))=:.2f}")
    print(f"\t{20*np.log10(np.max(abs(noise_dc)))=:.2f}")
    print (f"\t{20*np.log10(np.max(abs(total_dc)))=:.2f}")

def createWindow(inShape:tuple, plot=True):
    chwin = signal.windows.chebwin(inShape[1], 60)
    chwin_norm = chwin/np.mean(chwin)
    chwin_norm = chwin_norm.reshape((1, chwin.size))
    tmp = np.ones((inShape[0],1))
    chwin_norm_mat = tmp@chwin_norm
    if plot:
        plt.figure()
        plt.title("Window")
        plt.imshow(chwin_norm_mat)
        plt.xlabel("slow time")
        plt.ylabel("fast time")
        plt.colorbar()

    return chwin_norm_mat
