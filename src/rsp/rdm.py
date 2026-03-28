import numpy as np
from . import constants as c
from .rdm_helpers import plot_rtm, plot_rdm, plot_rdm_snr
from .rf_datacube import number_range_bins, range_axis, dataCube
from .rf_datacube import matchfilter, doppler_process
from .range_equation import noise_power
from .noise import unity_variance_complex_noise
from .rdm_helpers import add_returns, add_returns_snr, process_waveform_dict, create_window
from .__rdm_extras import noise_checks, check_expected_snr
from .pulse_doppler_radar import Radar


def gen(
    radar: Radar,
    waveform: dict,
    return_list: list,
    seed: int = 0,
    plot: bool = True,
    debug: bool = False,
    snr: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a Range-Doppler Map (RDM) for a single Coherent Processing Interval (CPI).

    Simulates received radar data for one or more targets moving at constant
    range rates and processes it to produce an RDM, accounting for radar system
    parameters, waveform characteristics, and noise.

    Args:
        radar: Radar system parameters. See
            :class:`rsp.pulse_doppler_radar.Radar` for required keys and units.
        waveform: Waveform definition dict. Use the factory functions
            (e.g. :func:`rsp.waveform.lfm_waveform`) to construct this.
        return_list: List of return dicts, each describing one simulated target.
            Each dict must have a ``'type'`` key (``'skin'`` or ``'memory'``)
            and a nested ``'target'`` dict with ``'range'``, ``'rangeRate'``,
            and (for skin) ``'rcs'``.
        seed: Random number generator seed for reproducibility. Defaults to 0.
        plot: If True, plots the final RDM. Defaults to True.
        debug: If True, plots intermediate processing steps and prints
            diagnostic statistics. Defaults to False.
        snr: If True, output amplitudes are normalised to SNR (voltage ratio).
            If False, output amplitudes are in Volts. Defaults to False.

    Returns:
        tuple: A four-element tuple ``(rdot_axis, r_axis, total_dc, signal_dc)``:

            - **rdot_axis** (*np.ndarray*): 1D range-rate (Doppler) axis [m/s].
            - **r_axis** (*np.ndarray*): 1D range axis [m].
            - **total_dc** (*np.ndarray*): 2D RDM including signal and noise,
              amplitude in Volts or SNR.
            - **signal_dc** (*np.ndarray*): 2D signal-only RDM, amplitude in
              Volts or SNR.
    """
    np.random.seed(seed)

    ########## Compute waveform and radar parameters ###############################################
    # Use normalized pulses, the time-bandwidth poduct is used for amp scaling
    process_waveform_dict(waveform, radar)

    ########## Create range axis for plotting ######################################################
    r_axis = range_axis(radar.sampRate, number_range_bins(radar.sampRate, radar.PRF))

    ########## Return ##############################################################################
    signal_dc = dataCube(radar.sampRate, radar.PRF, radar.Npulses)

    if snr:
        ### Direclty plot the RDM in SNR by way of the range equation ###
        # - The SNR is calculated at the initial range and does not change in time
        noise_dc = unity_variance_complex_noise(signal_dc.shape) / np.sqrt(radar.Npulses)
        add_returns_snr(signal_dc, waveform, return_list, radar)
    else:
        ### Determin scaling factors for max voltage ###
        rxVolt_noise = np.sqrt(
            c.RADAR_LOAD * noise_power(waveform["bw"], radar.noiseFactor, radar.opTemp)
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
        f_axis, r_axis = doppler_process(dc, radar.sampRate)

    ########## Plots and checks ####################################################################
    # calc rangeRate axis  #f = -2* fc/c Rdot -> Rdot = -c+f/ (2+fc)
    rdot_axis = -c.C * f_axis / (2 * radar.fcar)

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
