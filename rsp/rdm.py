import numpy as np
from . import constants as c
from .rdm_helpers import plot_rtm, plot_rdm
from .rf_datacube import number_range_bins, range_axis, dataCube
from .rf_datacube import matchfilter, doppler_process
from .waveform import process_waveform_dict
from .range_equation import noise_power, snr_range_eqn

from .rdm_helpers import noise_checks, create_window, check_expected_snr
from .rdm_helpers import add_returns


def gen(
    target: dict,
    radar: dict,
    waveform: dict,
    return_list: list,
    seed: int = 0,
    plot: bool = True,
    debug: bool = False,
    SNR: bool = False,
):
    """
    Generate a single CPI RDM for one target moving at a constant range rate.

    Parameters
    ----------
    target: dict with keys range, "rangeRate, rcs (all constant over the CPI)
    radar: dict with keys fcar, txPower, txGain, rxGain, opTemp, sampRate, noiseFactor, totalLosses, PRF
    waveform: dict with for waveform key types in ["uncoded", "barker", "random", "lfm"]
    returnInfo_list: list of dicts containing return types to place in the RDM, in ["skin", "memory"]

    Optional parameters:
    seed: int random seed
    plot: boolean to plot the final RDM
    debug: boolean to plot each step in building the RDM and print out statistics

    Returns
    -------
    rdot_axis: array of rangeRate axis [m/s]
    r_axis: range axisk [m]
    total_dc: RDM in Volts for noise + signal
    signal_dc: RDM in Volts for signal
    """

    # TODO: do I need to pass this seed to each function using random?
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
    SNR_onepulse = snr_range_eqn(
        radar["txPower"],
        radar["txGain"],
        radar["rxGain"],
        target["rcs"],
        c.C / radar["fcar"],
        target["range"],
        waveform["bw"],
        radar["noiseFactor"],
        radar["totalLosses"],
        radar["opTemp"],
        waveform["time_BW_product"],
    )

    SNR_volt = np.sqrt(SNR_onepulse / radar["Npulses"])

    ### Return  ##########################################
    signal_dc = dataCube(radar["sampRate"], radar["PRF"], radar["Npulses"])
    rxVolt_noise = np.sqrt(
        c.RADAR_LOAD * noise_power(waveform["bw"], radar["noiseFactor"], radar["opTemp"])
    )
    noise_dc = np.random.uniform(low=-1, high=1, size=signal_dc.shape) * rxVolt_noise

    add_returns(signal_dc, waveform, target, return_list, radar)
    # add_returns_snr(signal_dc, waveform, target, return_list, radar, rxVolt)

    total_dc = signal_dc + noise_dc  # adding after return keeps clean signal_dc for plotting

    if debug:
        plot_rtm(r_axis, signal_dc, "Noiseless RTM: unprocessed")

    ### Apply the match filter #############################
    for dc in [signal_dc, total_dc]:
        matchfilter(dc, waveform["pulse"], pedantic=True)

    if debug:
        plot_rtm(r_axis, signal_dc, "Noiseless RTM: match filtered")

    ### Doppler process ####################################
    # first create filter window and apply it
    chwin_norm_mat = create_window(signal_dc.shape, plot=False)
    total_dc = total_dc * chwin_norm_mat
    signal_dc = signal_dc * chwin_norm_mat

    # Doppler process datacubes
    for dc in [signal_dc, total_dc]:
        f_axis, r_axis = doppler_process(dc, radar["sampRate"])

    # calc rangeRate axis  #f = -2* fc/c Rdot -> Rdot = -c+f/ (2+fc)
    print("TODO: why PRF/fs ratio at end?")
    rdot_axis = -c.C * f_axis / (2 * radar["fcar"]) * radar["PRF"] / radar["sampRate"]

    if debug:
        plot_rdm(rdot_axis, r_axis, signal_dc, "Noiseless RDM")
        # SNR and noise checks
        check_expected_snr(radar, target, waveform, SNR_onepulse, SNR_volt)
        noise_checks(signal_dc, noise_dc, total_dc)

    if plot or debug:
        plot_rdm(rdot_axis, r_axis, total_dc, f"Total RDM for {waveform['type']}", cbarMin=0)

    return rdot_axis, r_axis, total_dc, signal_dc
