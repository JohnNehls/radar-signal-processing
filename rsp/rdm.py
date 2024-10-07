import numpy as np
from . import constants as c
from .rdm_helpers import plotRTM, plotRDM
from .rf_datacube import number_range_bins, range_axis, dataCube
from .rf_datacube import matchFilter, dopplerProcess
from .waveform import process_waveform_dict
from .range_equation import snr_rangeEquation
from .rdm_helpers import addReturns, noiseChecks, createWindow, checkExpectedSNR


def rdm_gen(
    target: dict,
    radar: dict,
    waveform: dict,
    return_list: list,
    seed: int = 0,
    plot=True,
    debug=False,
):
    """
    Genearat a single CPI RDM for one target moving at a constant range rate.

    Parameters
    ----------
    target: dict with keys range, "rangeRate, rcs (all constant over the CPI)
    radar: dict with keys fcar, txPower, txGain, rxGain, opTemp, sampRate, noiseFig, totalLosses, PRF
    waveform: dict with for wvform key types in ["uncoded", "barker", "random", "lfm"]
    returnInfo_list: list of dicts containing return types to place in the RDM, in ["skin", "memory"]

    Optional parameters
    seed: int random seed for random module
    plot: boolean to plot the final RDM
    debug: boolean to plot each step in building the RDM and print out statistics

    Returns
    -------
    rdot_axis: array of rangeRate axis [m/s]
    r_axis: range axisk [m]
    total_dc: RDM in Volts for noise + signal
    signal_dc: RDM in Volts for signal
    """

    np.random.seed(seed)

    ### Compute waveform and radar parameters ##############
    # Use normalized pulses, the time-bandwidth poduct is used for amp scaling
    process_waveform_dict(waveform, radar)
    radar["Npulses"] = int(np.ceil(radar["dwell_time"] * radar["PRF"]))

    ### Create range axis for plotting #####################
    r_axis = range_axis(radar["sampRate"], number_range_bins(radar["sampRate"], radar["PRF"]))

    ### Determin scaling factor for SNR ####################
    # - Motivation is to  direclty plot the RDM in SNR by way of the range equation
    # - The SNR is calculated at the initial range and does not change in time
    SNR_onepulse = snr_rangeEquation(
        radar["txPower"],
        radar["txGain"],
        radar["rxGain"],
        target["rcs"],
        c.C / radar["fcar"],
        target["range"],
        waveform["bw"],
        radar["noiseFig"],
        radar["totalLosses"],
        radar["opTemp"],
        waveform["time_BW_product"],
    )

    SNR_volt = np.sqrt(SNR_onepulse / radar["Npulses"])

    ### Return  ##########################################
    signal_dc = dataCube(radar["sampRate"], radar["PRF"], radar["Npulses"])
    noise_dc = dataCube(radar["sampRate"], radar["PRF"], radar["Npulses"], noise=True)
    addReturns(signal_dc, waveform, target, return_list, radar, SNR_volt)
    total_dc = signal_dc + noise_dc  # adding after return keeps clean signal_dc for plotting

    if debug:
        plotRTM(r_axis, signal_dc, "Noiseless RTM: unprocessed")

    ### Apply the match filter #############################
    for dc in [signal_dc, total_dc]:
        matchFilter(dc, waveform["pulse"], pedantic=True)

    if debug:
        plotRTM(r_axis, signal_dc, "Noiseless RTM: match filtered")

    ### Doppler process ####################################
    # first create filter window and apply it
    chwin_norm_mat = createWindow(signal_dc.shape, plot=False)
    total_dc = total_dc * chwin_norm_mat
    signal_dc = signal_dc * chwin_norm_mat

    # Doppler process datacubes
    for dc in [signal_dc, total_dc]:
        f_axis, r_axis = dopplerProcess(dc, radar["sampRate"], radar["PRF"])

    # calc rangeRate axis  #f = -2* fc/c Rdot -> Rdot = -c+f/ (2+fc)
    print("TODO Why PRF/fs ratio at end?")
    rdot_axis = -c.C * f_axis / (2 * radar["fcar"]) * radar["PRF"] / radar["sampRate"]

    if debug:
        plotRDM(rdot_axis, r_axis, signal_dc, "Noiseless RDM")
        # SNR and noise checks
        checkExpectedSNR(radar, target, waveform, SNR_onepulse, SNR_volt)
        noiseChecks(signal_dc, noise_dc, total_dc)

    if plot or debug:
        plotRDM(rdot_axis, r_axis, total_dc, f"Total RDM for {waveform['type']}", cbarMin=0)

    return rdot_axis, r_axis, total_dc, signal_dc
