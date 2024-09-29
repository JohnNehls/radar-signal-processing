#!/usr/bin/env python

import numpy as np
from scipy import fft
from scipy import signal
import matplotlib.pyplot as plt
from constants import C
from rdm_helpers import plotRDM, plotRTM
from pulse_doppler_radar import range_unambiguous
from rf_datacube import calc_number_range_bins, calc_range_axis, create_dataCube
from rf_datacube import applyMatchFilterToDataCube, dopplerProcess_dataCube
from waveform import process_waveform_dict
from range_equation import snr_rangeEquation, snr_rangeEquation_CP
from rdm_helpers import addSkin, addMemory


def rdm_gen(tgtInfo: dict, radar: dict, wvf: dict, returnInfo: dict,
            seed=None, plotSteps=False):
    """
    Genearat a CPI RDM for single target moving at a constant range rate.

    Parameters
    ----------
    tgtInfo: dict holding range, rangeRate, and rcs (range rate and rcs constant for CPI)
    radar: dict holding fcar, txPower, txGain, rxGain, opTemp, sampRate, noiseFig, totalLosses, PRF
    wvf: string for wvform types : "uncoded" "barker" "random" "lfm"
    Npulses: number of puleses in the CPI

    Returns
    -------
    rdot_axis: array of rangeRate axis [m/s]
    r_axis: range axisk [m]
    total_dc: RDM in Volts for noise + signal
    signal_dc: RDM in Volts for signal
    noise_dc: RDM in Volts for noise
    """

    ### set random seed ####################################
    if seed != None:
        print (f"{seed=}")
        np.random.seed(seed)

    ### Compute waveform and radar parameters ##############
    # Use normalized pulses the time-bandwidth poduct is used for amp scaling
    process_waveform_dict(wvf, radar)
    radar["Npulses"] = int(np.ceil(radar["dwell_time"] * radar["PRF"]))

    ### Create range and time axes #########################
    t_slow_axis = np.arange(radar["Npulses"])*1/radar["PRF"]
    NrangeBins = calc_number_range_bins(radar["sampRate"], radar["PRF"])
    r_axis = calc_range_axis(radar["sampRate"], NrangeBins)
    t_fast_axis = 2*r_axis/C

    ### Determin scaling factor for SNR ####################
    # Motivation: direclty plot the RDM in SNR by way of the range equation
    # notes:
    # - The SNR is calculated at the initial range and does not change in time

    # SNR for one pulse
    SNR1 = snr_rangeEquation(radar["txPower"], radar["txGain"], radar["rxGain"],
                             tgtInfo["rcs"], C/radar["fcar"], tgtInfo["range"],
                             wvf["bw"], radar["noiseFig"], radar["totalLosses"],
                             radar["opTemp"], wvf["time_BW_product"])

    SNR_volt = np.sqrt(SNR1/radar["Npulses"])

    # calculate the expected SNR
    SNR_expected = snr_rangeEquation_CP(radar["txPower"], radar["txGain"],
                                        radar["rxGain"], tgtInfo["rcs"],
                                        C/radar["fcar"], tgtInfo["range"], wvf["bw"],
                                        radar["noiseFig"], radar["totalLosses"],
                                        radar["opTemp"], radar["Npulses"], wvf["time_BW_product"])
    print (f"SNR Check: ")
    print(f"\t{10*np.log10(SNR1)=:.2f}")
    print(f"\t{SNR_volt=:.1e}\n\t{SNR_expected=:.1e}")
    print(f"\t{10*np.log10(SNR_expected)=:.2f}")

    ### Return  ##########################################
    signal_dc = create_dataCube(radar["sampRate"], radar["PRF"], radar["Npulses"])
    #Range and range rate of the target
    # Currently takes in constant range rate
    tgt_range_ar = tgtInfo["range"] + tgtInfo["rangeRate"]*t_slow_axis

    ### Find first response pulse location #################
    # firstEchoBin = int(tgtInfo["range"]/range_unambiguous(radar ["PRF"]))

    ## Skin : place pulse at range index and apply phase ###########################
    if returnInfo["type"] == "skin":
        addSkin(signal_dc, wvf, radar, tgt_range_ar, r_axis, SNR_volt)

    ## Memory : place pulse at range index and apply phase #############################
    elif returnInfo["type"] == "memory":
        addMemory(signal_dc, wvf, tgtInfo, radar, returnInfo, t_slow_axis, t_fast_axis, SNR_volt)
    else:
        print(f"{returnInfo['type']=} not known, no return added.")

    ### Create noise and total datacube ####################
    noise_dc = create_dataCube(radar["sampRate"], radar["PRF"], radar["Npulses"], noise=True)

    print(f"\n5.3.2 noise check: {np.var(fft.fft(noise_dc, axis=1))=: .4f}")

    total_dc = signal_dc + noise_dc

    if plotSteps:
        plotRTM(r_axis, signal_dc, f"SIGNAL: unprocessed {wvf['type']}")

    ### Apply the match filter #############################
    applyMatchFilterToDataCube(signal_dc, wvf["pulse"])
    applyMatchFilterToDataCube(noise_dc, wvf["pulse"])
    applyMatchFilterToDataCube(total_dc, wvf["pulse"])

    if plotSteps:
        plotRTM(r_axis, signal_dc, f"SIGNAL: match filtered {wvf['type']}")
        # plotRTM(r_axis, noise_dc,   f"NOISE: match filtered {wvf['type']}")
        # plotRTM(r_axis, total_dc,   f"TOTAL: match filtered {wvf['type']}")

    ### Doppler process ####################################
    # create filter window
    chwin = signal.windows.chebwin(radar["Npulses"], 60)
    chwin_norm = chwin/np.mean(chwin)
    chwin_norm = chwin_norm.reshape((1, chwin.size))
    tmp = np.ones((total_dc.shape[0],1))
    chwin_norm_mat = tmp@chwin_norm

    # apply filter window
    total_dc = total_dc*chwin_norm_mat
    signal_dc = signal_dc*chwin_norm_mat

    # if plotSteps:
        # plotRTM(r_axis, signal_dc, f"SIGNAL: mf & windowed {wvf["type"]}")
        # plotRTM(r_axis, total_dc,   f"TOTAL: mf & windowed {wvf["type"]}")

    # doppler process in place
    f_axis, r_axis = dopplerProcess_dataCube(signal_dc, radar["sampRate"], radar["PRF"])
    _, _           = dopplerProcess_dataCube(noise_dc,  radar["sampRate"], radar["PRF"])
    _, _           = dopplerProcess_dataCube(total_dc,  radar["sampRate"], radar["PRF"])

    # calc rangeRate axis
    #f = -2* fc/c Rdot -> Rdot = -c+f/ (2+fc)
    #TODO WHY PRF/fs ratio at end??!?!
    rdot_axis = -C*f_axis/(2*radar["fcar"])*radar["PRF"]/radar["sampRate"]


    ### Verify SNR and noise ###################################
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

    return rdot_axis, r_axis, total_dc, signal_dc, noise_dc

def main():
    plt.close('all')
    ################################################################################
    # Example below is ephemeral and used for debugging new features
    # - Main examples are in the ./example_rdm directory
    ################################################################################

    # Function inputs ########################################################################
    bw = 10e6

    tgtInfo = {"range": 3.5e3,
               "rangeRate": 0.5e3,
               "rcs" : 10}

    radar = {"fcar" : 10e9,
             "txPower": 1e3,
             "txGain" : 10**(30/10),
             "rxGain" : 10**(30/10),
             "opTemp": 290,
             "sampRate": 2*bw,
             "noiseFig": 10**(8/10),
             "totalLosses" : 10**(8/10),
             "PRF": 200e3,
             "dwell_time" : 2e-3}

    wvf = {"type" : None} # noise test

    wvf = {"type": "uncoded",
           "bw" : bw}

    wvf = {"type" : "barker",
           "nchips" : 13,
           "bw" : bw}

    # # wvf = {"type": "random",
    #        "nchips" : 13,
    #        "bw" : bw}

    wvf = {"type": "lfm",
           "bw" : bw,
           "T": 10/40e6,
           'chirpUpDown': 1}

    returnInfo = {"type" : "skin"}

    returnInfo = {"type" : "memory",
                  "rdot_delta" : 0.5e3,
                  "rdot_offset" : 0.1e3,
                  "range_offset" : -0.0e3,
                  }


    plotsteps = True


    ## Call function ###############################################################
    rdot_axis, r_axis, total_dc, signal_dc, noise_dc = rdm_gen(tgtInfo, radar,
                                                               wvf,
                                                               returnInfo,
                                                               seed=0,
                                                               plotSteps=plotsteps)

    ## Plot outputs ################################################################
    plotRDM(rdot_axis, r_axis, signal_dc, f"SIGNAL: dB doppler processed match filtered {wvf['type']}")
    plotRDM(rdot_axis, r_axis, total_dc,
            f"TOTAL: dB doppler processed match filtered {wvf['type']}", cbarRange=False)
    # plotRDM(rdot_axis, r_axis, noise_dc, f"NOISE: dB doppler processed match filtered {wvf['type']}")

    plt.show()


if __name__ == "__main__":
    main()
