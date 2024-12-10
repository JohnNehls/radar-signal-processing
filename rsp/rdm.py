import numpy as np
from scipy import interpolate
from . import constants as c
from .rdm_helpers import plot_rtm, plot_rdm, plot_rdm_snr
from .rf_datacube import number_range_bins, range_axis, dataCube
from .rf_datacube import matchfilter, doppler_process
from .waveform import process_waveform_dict
from .range_equation import noise_power
from .noise import unity_variance_complex_noise
from .rdm_helpers import noise_checks, create_window, check_expected_snr
from .rdm_helpers import add_returns, add_returns_snr
import rsp.uniform_linear_arrays as ula
import matplotlib.pyplot as plt


def gen(
    radar: dict,
    waveform: dict,
    return_list: list,
    seed: int = 0,
    plot: bool = True,
    debug: bool = False,
    snr: bool = False,
    array_positions: list = 0
):
    """
    Generate a single CPI RDM for one target moving at a constant range rate.

    Parameters
    ----------
    radar: dict with fcar, txPower, txGain, rxGain, opTemp, sampRate, noiseFactor, totalLosses, PRF
    waveform: dict for waveform types in ["uncoded", "barker", "random", "lfm"]
    returnInfo_list: list of dicts containing return types to place in the RDM, in ["skin", "memory"]

    Optional parameters:
    seed: int random seed
    plot: boolean plot the final RDM
    debug: boolean plot each step in building the RDM and print out statistics
    snr: boolean create the RDM in SNR for the skin retrun (memory return is then notional)

    Returns
    -------
    rdot_axis: array of rangeRate axis [m/s]
    r_axis: range axisk [m]
    total_dc: RDM in Volts for noise + signal
    signal_dc: RDM in Volts for signal
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

    ### Alter the signal due to linear array position #####################
    # - These are experimental at this time
    tgt_angle = 70
    assert abs(tgt_angle) < 90

    signal_dc_ula_list = []
    for pos in array_positions:
        # apply phase shift from carrier 
        signal_dc_sv = signal_dc * ula.steering_vector(pos, tgt_angle)

        # apply time shift to baseband signal from array position from zero        
        tmp_signal = signal_dc_sv.T.flatten() 
        shifted_signal = ula.apply_timeshift_due_to_element_position(tmp_signal, radar["sampRate"], pos, tgt_angle)
        signal_dc_shift = shifted_signal.reshape(tuple(reversed(signal_dc_sv.shape))).T
        
        signal_dc_ula_list.append(signal_dc_shift)
        # signal_dc_ar.append(signal_dc_sv)   # IGNORE timeShift

    total_dc = signal_dc + noise_dc  # adding after return keeps clean signal_dc for plotting    

    # plot imaginary since the real parts are identical for symetric linear arrays (Cos)
    plt.figure()
    plt.title("Imaginary component of signal")
    plt.plot(tmp_time, np.imag(signal_dc_ula_list[0].T.flatten()), label="neg position")
    plt.plot(tmp_time, np.imag(signal_dc_ula_list[1].T.flatten()), '--', label="pos position")
    plt.legend()
    

    # plot imaginary since the real parts are identical for symetric linear arrays (Cos)
    plt.figure()
    plt.title("Angle difference")
    a = signal_dc_ula_list[0].T.flatten()
    b = signal_dc_ula_list[1].T.flatten()
    theta = np.arccos( a * np.conjugate(b) / (abs(a) * abs(b)))
    plt.plot(tmp_time, theta )
    # plt.plot(tmp_time, np.angle(signal_dc_ar[0].T.flatten()), label="neg position")
    # plt.plot(tmp_time, np.angle(signal_dc_ar[1].T.flatten()), '--', label="pos position")
    plt.legend()
    plt.show()


    rdm_list = [signal_dc, total_dc]

    if debug:
        plot_rtm(r_axis, signal_dc, "Noiseless RTM: unprocessed")

    ########## Apply the match filter ##############################################################
    for dc in rdm_list:
        matchfilter(dc, waveform["pulse"], pedantic=True)

    if debug:
        plot_rtm(r_axis, signal_dc, "Noiseless RTM: match filtered")

    ########### Doppler process ####################################################################
    # First create filter window and apply it
    chwin_norm_mat = create_window(signal_dc.shape, plot=False)
    for dc in rdm_list:
        dc = dc * chwin_norm_mat

    # Doppler process datacubes
    for dc in rdm_list:
        f_axis, r_axis = doppler_process(dc, radar["sampRate"])

    ########## Plots and checks ####################################################################
    # calc rangeRate axis  #f = -2* fc/c Rdot -> Rdot = -c+f/ (2+fc)
    rdot_axis = -c.C * f_axis / (2 * radar["fcar"])

    if debug:
        if snr:
            plot_rdm_snr(rdot_axis, r_axis, signal_dc, "Noiseless RDM", cbarMin=0)
            noise_checks(signal_dc, noise_dc, total_dc)
        else:
            plot_rdm(rdot_axis, r_axis, signal_dc, "Noiseless RDM")
    if plot or debug:
        if snr:
            plot_rdm_snr(
                rdot_axis, r_axis, total_dc, f"Total SNR RDM for {waveform['type']}", cbarMin=0
            )
            # if debug:
            check_expected_snr(radar, return_list[0]["target"], waveform)  # first return item
        else:
            plot_rdm(rdot_axis, r_axis, total_dc, f"Total RDM for {waveform['type']}")

    return rdot_axis, r_axis, total_dc, signal_dc
