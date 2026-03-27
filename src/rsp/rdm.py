import numpy as np
from . import constants as c
from .rdm_helpers import plot_rtm, plot_rdm, plot_rdm_snr
from .rf_datacube import number_range_bins, range_axis, dataCube
from .rf_datacube import matchfilter, doppler_process
from .range_equation import noise_power
from .noise import unity_variance_complex_noise
from .rdm_helpers import add_returns, add_returns_snr, process_waveform_dict, create_window
from .__rdm_extras import noise_checks, check_expected_snr


def gen(
    radar: dict,
    waveform: dict,
    return_list: list,
    seed: int = 0,
    plot: bool = True,
    debug: bool = False,
    snr: bool = False,
):
    """Generate a Range-Doppler Map (RDM) for a single Coherent Processing Interval (CPI).

    This function simulates the received radar data for one or more targets
    moving at constant range rates and processes it to produce an RDM. It
    accounts for radar system parameters, waveform characteristics, and noise.

    Parameters
    ----------
    radar : dict
        A dictionary containing the radar system parameters. Expected keys are:
        'fcar' : float
            Carrier frequency in Hertz (Hz).
        'txPower' : float
            Transmit power in Watts (W).
        'txGain' : float
            Transmit antenna gain in decibels (dB).
        'rxGain' : float
            Receive antenna gain in decibels (dB).
        'opTemp' : float
            Operating temperature in Kelvin (K).
        'sampRate' : float
            Sampling rate in Hertz (Hz).
        'noiseFactor' : float
            Receiver noise factor in decibels (dB).
        'totalLosses' : float
            Total system losses in decibels (dB).
        'PRF' : float
            Pulse Repetition Frequency in Hertz (Hz).
    waveform : dict
        A dictionary describing the transmitted waveform. Must contain a 'type'
        key, with other keys dependent on the type.
        'type' : {"uncoded", "barker", "random", "lfm"}
            The type of waveform modulation.
    return_list : list of dict
        A list where each dictionary defines a target return to be simulated.
        Each dictionary should contain details about the return, such as:
        'type' : {"skin", "memory"}
            The type of radar return.
        'rcs' : float
            Radar Cross Section in square meters (m^2).
        'range' : float
            Initial target range in meters (m).
        'range_rate' : float
            Target's radial velocity in meters per second (m/s).
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
        Default is 0.
    plot : bool, optional
        If True, generates a plot of the final Range-Doppler Map.
        Default is True.
    debug : bool, optional
        If True, plots intermediate steps of the RDM generation and prints
        diagnostic statistics. Default is False.
    snr : bool, optional
        If True, the output RDM amplitudes are normalized to the Signal-to-Noise
        Ratio (SNR) of the primary skin return. If False, amplitudes are in Volts.
        Default is False.

    Returns
    -------
    tuple
        A tuple containing the following four numpy arrays:
        rdot_axis : numpy.ndarray
            1D array representing the range-rate (Doppler) axis of the RDM in
            meters per second (m/s).
        r_axis : numpy.ndarray
            1D array representing the range axis of the RDM in meters (m).
        total_dc : numpy.ndarray
            2D array representing the complete RDM, including both
            signal and noise, with amplitude in Volts or SNR.
        signal_dc : numpy.ndarray
            2D array representing the signal-only RDM, with
            amplitude in Volts or SNR.
    """
    np.random.seed(seed)

    ########## Compute waveform and radar parameters ###############################################
    # Use normalized pulses, the time-bandwidth poduct is used for amp scaling
    process_waveform_dict(waveform, radar)
    radar["Npulses"] = int(np.ceil(radar["dwell_time"] * radar["PRF"]))

    ########## Create range axis for plotting ######################################################
    r_axis = range_axis(radar["sampRate"], number_range_bins(radar["sampRate"], radar["PRF"]))

    ########## Return ##############################################################################
    signal_dc = dataCube(radar["sampRate"], radar["PRF"], radar["Npulses"])

    if snr:
        ### Direclty plot the RDM in SNR by way of the range equation ###
        # - The SNR is calculated at the initial range and does not change in time
        noise_dc = unity_variance_complex_noise(signal_dc.shape) / np.sqrt(radar["Npulses"])
        add_returns_snr(signal_dc, waveform, return_list, radar)
    else:
        ### Determin scaling factors for max voltage ###
        rxVolt_noise = np.sqrt(
            c.RADAR_LOAD * noise_power(waveform["bw"], radar["noiseFactor"], radar["opTemp"])
        )
        noise_dc = np.random.uniform(low=-1, high=1, size=signal_dc.shape) * rxVolt_noise
        add_returns(signal_dc, waveform, return_list, radar)

    total_dc = signal_dc + noise_dc  # adding after return keeps clean signal_dc for plotting

    if debug:
        plot_rtm(r_axis, signal_dc, "Noiseless RTM: unprocessed")

    # list of datacubes to process in the following steps
    rdm_list = [signal_dc, total_dc]

    ########## Apply the match filter ##############################################################
    for dc in rdm_list:
        matchfilter(dc, waveform["pulse"], pedantic=True)

    if debug:
        plot_rtm(r_axis, signal_dc, "Noiseless RTM: match filtered")

    ########### Doppler process ####################################################################
    # First create filter window and apply it
    chwin_norm_mat = create_window(signal_dc.shape, plot=False)
    for dc in rdm_list:
        dc *= chwin_norm_mat

    # Doppler process datacubes
    for dc in rdm_list:
        f_axis, r_axis = doppler_process(dc, radar["sampRate"])

    ########## Plots and checks ####################################################################
    # calc rangeRate axis  #f = -2* fc/c Rdot -> Rdot = -c+f/ (2+fc)
    rdot_axis = -c.C * f_axis / (2 * radar["fcar"])

    if debug:
        if snr:
            plot_rdm_snr(rdot_axis, r_axis, signal_dc, "Noiseless RDM", cbar_min=0)
            noise_checks(signal_dc, noise_dc, total_dc)
        else:
            plot_rdm(rdot_axis, r_axis, signal_dc, "Noiseless RDM")
    if plot or debug:
        if snr:
            plot_rdm_snr(
                rdot_axis, r_axis, total_dc, f"Total SNR RDM for {waveform['type']}", cbar_min=0
            )
            # if debug:
            check_expected_snr(radar, return_list[0]["target"], waveform)  # first return item
        else:
            plot_rdm(rdot_axis, r_axis, total_dc, f"Total RDM for {waveform['type']}")

    return rdot_axis, r_axis, total_dc, signal_dc
