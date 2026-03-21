import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from . import constants as c
from . import waveform as wvf
from .waveform_helpers import add_waveform_at_index
from .utilities import phase_negpi_pospi, zero_to_smallest_float
from .range_equation import snr_range_eqn, signal_range_eqn
from . import vbm


def plot_rtm(r_axis, data, title):
    """Plots the magnitude and phase of a range-time matrix (RTM).

    The RTM shows radar data before Doppler processing, with range on one
    axis and pulse number (slow-time) on the other.

    Args:
        r_axis (np.ndarray): 1D array of range values in meters.
        data (np.ndarray): 2D complex array representing the RTM.
                           Shape should be (num_range_bins, num_pulses).
        title (str): The title for the plot.

    Returns:
        None
    """
    pulses = range(data.shape[1])
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)
    p = ax[0].pcolormesh(pulses, r_axis * 1e-3, abs(data))
    ax[0].set_xlabel("pulse number")
    ax[0].set_ylabel("range [km]")
    ax[0].set_title("magnitude")
    fig.colorbar(p)
    ax[1].pcolormesh(pulses, r_axis * 1e-3, np.angle(data))
    ax[1].set_xlabel("pulse number")
    ax[1].set_ylabel("range [km]")
    ax[1].set_title("phase")
    fig.tight_layout()


def plot_rdm(rdot_axis, r_axis, data, title, cbarMin=-100, volt2dbm=True):
    """Plots a range-Doppler matrix (RDM).

    The RDM shows radar data after pulse compression and Doppler processing.

    Args:
        rdot_axis (np.ndarray): 1D array of range-rate values in m/s.
        r_axis (np.ndarray): 1D array of range values in meters.
        data (np.ndarray): 2D complex array representing the RDM.
        title (str): The title for the plot.
        cbarMin (float, optional): The minimum value for the color bar.
                                   Defaults to -100.
        volt2dbm (bool, optional): If True, converts the data from volts
                                   to dBm for plotting. Defaults to True.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects of the plot.
    """
    data = abs(data)  # complex -> real

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set_xlabel("range rate [km/s]")
    ax.set_ylabel("range [km]")
    if volt2dbm:
        zero_to_smallest_float(data)  # needed for signal_dc plots
        data = 20 * np.log10(data / np.sqrt(1e-3 * c.RADAR_LOAD))
        p = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, data)
        p.set_clim(cbarMin, data.max())
        cbar = fig.colorbar(p)
        cbar.set_label("Power [dBm]")
    else:
        p = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, data)
        p.set_clim(cbarMin, data.max())
        cbar = fig.colorbar(p)
        cbar.set_label("Power [W]")

    fig.tight_layout()

    return fig, ax


def plot_rdm_snr(rdot_axis, r_axis, data, title, cbarMin=0, volt2db=True):
    """Plots a range-Doppler matrix in terms of Signal-to-Noise Ratio (SNR).

    Args:
        rdot_axis (np.ndarray): 1D array of range-rate values in m/s.
        r_axis (np.ndarray): 1D array of range values in meters.
        data (np.ndarray): 2D array representing the RDM in SNR (linear units).
        title (str): The title for the plot.
        cbarMin (float, optional): The minimum value for the color bar.
                                   Defaults to 0.
        volt2db (bool, optional): If True, converts the SNR data to dB.
                                  Defaults to True.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects of the plot.
    """
    data = abs(data)  # complex -> real
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set_xlabel("range rate [km/s]")
    ax.set_ylabel("range [km]")
    if volt2db:
        zero_to_smallest_float(data)  # needed for signal_dc plots
        data = 20 * np.log10(data)
    p = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, data)
    print("cbarmin")
    p.set_clim(cbarMin, data.max())
    cbar = fig.colorbar(p)
    if volt2db:
        cbar.set_label("SNR [dB]")
    else:
        cbar.set_label("SNR")

    fig.tight_layout()

    return fig, ax


def add_skin(datacube, wvf: dict, tgtInfo: dict, radar: dict, return_magnitude: float):
    """Adds a direct radar reflection (skin return) from a target to the datacube.

    This function simulates the signal received by the radar after it reflects
    off a target. The function calculates the time delay and phase shift for each
    pulse and adds the appropriately modified waveform to the datacube.

    The datacube is modified in place.

    Args:
        datacube (np.ndarray): 2D complex array to which the return is added.
        wvf (dict): Dictionary containing waveform parameters, including the
                    'pulse' (waveform sample array) and 'pulse_width'.
        tgtInfo (dict): Dictionary with target information, including 'range',
                        'rangeRate', and optionally 'sv' (steering vector).
        radar (dict): Dictionary with radar parameters, such as 'sampRate',
                      'Npulses', 'PRF', and 'fcar'.
        return_magnitude (float): The voltage or SNR amplitude of the return.
    """
    # time and range arrays
    time_ar = np.arange(datacube.size) * 1 / radar["sampRate"]  # time of all samples in CPI
    t_slow_axis = np.arange(radar["Npulses"]) * 1 / radar["PRF"]  # time when pulses sent

    tgt_range_ar = tgtInfo["range"] + tgtInfo["rangeRate"] * t_slow_axis  # tgt range at pulse send
    twoWay_time_delay_ar = 2 * tgt_range_ar / c.C  # time of travel from radar to tgt and back
    pulse_return_time = t_slow_axis + twoWay_time_delay_ar  # time pulses return to radar
    twoWay_phase_ar = -2 * c.PI * radar["fcar"] * twoWay_time_delay_ar  # Phase added due to

    ## pulses timed from their start not their center, we compensate with pw/2 range offset
    time_pw_offset = wvf["pulse_width"] / 2

    # Due to the time axis being the non-continuous (slow) axis, we most do some transposing
    tmpSignal = datacube.T.flatten()

    for i in range(radar["Npulses"]):
        # TODO is this how these should be binned? Should they be interpolated onto grid?
        # - (-1) below added to make return end up in correct range bin (lfm still off)
        #   - see ../tests/5_1_skin_in_correct_rangedoppler_bin.py for details
        timeIndex = np.argmin(abs(time_ar - pulse_return_time[i] + time_pw_offset))
        if timeIndex > 0:
            timeIndex -= 1
        pulse = return_magnitude * wvf["pulse"] * np.exp(1j * twoWay_phase_ar[i])
        pulse *= tgtInfo.get("sv", 1)  # account for linear array position

        if timeIndex < datacube.size:  # else pulse is in next CPI
            add_waveform_at_index(tmpSignal, pulse, timeIndex)

    tmpSignal = tmpSignal.reshape(tuple(reversed(datacube.shape))).T

    datacube[:] = tmpSignal[:]


def add_memory(datacube, wvf: dict, radar: dict, returnInfo: dict, return_magnitude: float):
    """Adds a notional memory-based electronic attack (EA) return to the datacube.

    This function simulates a Digital Radio Frequency Memory (DRFM) jammer that
    records an incoming pulse and re-transmits it with modifications to deceive
    the radar (e.g., range or Doppler offsets, Velocity Bin Masking).

    The datacube is modified in place.

    Args:
        datacube (np.ndarray): 2D complex array to which the return is added.
        wvf (dict): Dictionary containing waveform parameters.
        radar (dict): Dictionary with radar parameters.
        returnInfo (dict): Dictionary with EA and target information, including
                           'target', 'delay', 'range_offset', 'rdot_offset',
                           'rdot_delta' (for VBM).
        return_magnitude (float): The voltage or SNR amplitude of the return.
    """
    print("TODO: verify add_memory account for linear array position")
    print("Note: memory return amplitudes are notional")
    if "rcs" in returnInfo["target"]:
        print("Note: returnInfo['target']['rcs'] ignored for 'memory' returns")

    # time and range arrays
    time_ar = np.arange(datacube.size) * 1 / radar["sampRate"]  # time of all samples in CPI
    t_slow_axis = np.arange(radar["Npulses"]) * 1 / radar["PRF"]  # time when pulses sent

    # tgt range at pulse send
    tgt_range_ar = returnInfo["target"]["range"] + returnInfo["target"]["rangeRate"] * t_slow_axis
    oneWay_time_delay_ar = tgt_range_ar / c.C  # time of travel from radar to tgt
    # TODO this should be changed to when pod transmits, not when pulse was transmitted
    pulse_return_time = t_slow_axis + 2 * oneWay_time_delay_ar  # time pulses return to radar
    oneWay_phase_ar = -2 * c.PI * radar["fcar"] * oneWay_time_delay_ar  # Phase added due to

    ## pulses timed from their start not their center, we compensate with pw/2 range offset
    time_pw_offset = wvf["pulse_width"] / 2

    # Make output offset from skin return #############################################
    # - remove x2 for absolute rdot
    f_rdot = 2 * radar["fcar"] / c.C * returnInfo.get("rdot_offset", 0)

    # Achieve Velocity Bin Masking (VBM) by adding pahse in slow time #################
    if "rdot_delta" in returnInfo.keys():
        # there are several methods implemented, lfm is best, see vbm.py
        vbm_noise_function = returnInfo.get("vbm_noise_function", vbm._lfm_phase)
        slowtime_noise = vbm.slowtime_noise(
            radar["Npulses"],
            radar["fcar"],
            returnInfo["rdot_delta"],
            radar["PRF"],
            noiseFun=vbm_noise_function,
        )

    else:
        slowtime_noise = np.ones(radar["Npulses"])  # default if no VBM

    # Delay the return ################################################################
    # - can be negative, default is zero
    delay = returnInfo.get("delay", 0)
    delay += 2 * returnInfo.get("range_offset", 0) / c.C

    stored_pulse = 0
    stored_angle = 0  # initialize to stop lsp from complaining

    # Due to the time axis being the non-continuous (slow) axis, we most do some transposing
    tmpSignal = datacube.T.flatten()

    for i in range(radar["Npulses"]):
        # pulse recieved by the EW system
        recieved_pulse = wvf["pulse"] * np.exp(1j * oneWay_phase_ar[i])

        # Store first pulse and wait for next pulse
        if i == 0:
            stored_pulse = recieved_pulse
            continue

        # Calculate 1-way phase difference between first two pulses
        # - in a more complicated system, we'd look at the phase diff of max of match filter
        if i == 1:
            stored_angle = np.angle(recieved_pulse) - np.angle(stored_pulse)
            stored_angle = phase_negpi_pospi(stored_angle)
            stored_angle = np.mean(stored_angle)

        # Create base pulse
        # - TODO set amplitude base on pod parameters
        pulse = return_magnitude * stored_pulse
        pulse *= returnInfo["target"].get("sv", 1)  # account for linear array position

        # add slowtime noise (VBM)
        pulse = pulse * slowtime_noise[i]

        # add stored pulse difference rdot
        pulse = pulse * (np.exp(1j * i * stored_angle))

        # add rdot offset
        pulse = pulse * (np.exp(-1j * i * 2 * c.PI * f_rdot / radar["PRF"]))

        # add 1-way propagation phase back to radar
        pulse = pulse * np.exp(1j * oneWay_phase_ar[i])

        # TODO is this how these should be binned? Should they be interpolated onto grid?
        # - Like skin index, -1 was added to better match expected results
        #   - see ../tests/5_2_memory_in_correct_rangedoppler_bin.py for details
        timeIndex = np.argmin(abs(time_ar - pulse_return_time[i] - delay + time_pw_offset))
        if timeIndex > 0:
            timeIndex -= 1
        if timeIndex < datacube.size:  # else pulse is in next CPI
            add_waveform_at_index(tmpSignal, pulse, timeIndex)

    tmpSignal = tmpSignal.reshape(tuple(reversed(datacube.shape))).T

    datacube[:] = tmpSignal[:]


def create_window(inShape: tuple, plot=True):
    """Creates a 2D Dolph-Chebyshev window for sidelobe reduction.

    The window is applied along the slow-time (pulse) dimension to reduce
    Doppler sidelobes.

    Args:
        inShape (tuple): The desired shape of the output window (num_range_bins,
                         num_pulses).
        plot (bool, optional): If True, displays a plot of the generated
                               window. Defaults to True.

    Returns:
        np.ndarray: The 2D window matrix of shape `inShape`.
    """
    assert len(inShape) == 2
    # more window options are used in the window comparison in examples/tests/
    chwin = signal.windows.chebwin(inShape[1], 60)
    chwin_norm = chwin / np.mean(chwin)
    chwin_norm = chwin_norm.reshape((1, chwin.size))
    tmp = np.ones((inShape[0], 1))
    chwin_norm_mat = tmp @ chwin_norm
    if plot:
        plt.figure()
        plt.title("Window")
        plt.imshow(chwin_norm_mat)
        plt.xlabel("slow time")
        plt.ylabel("fast time")
        plt.colorbar()

    return chwin_norm_mat


def skin_snr_amplitude(radar, target, waveform):
    """Calculates the expected SNR amplitude for a skin return.

    Uses the radar range equation to determine the signal-to-noise ratio
    for a single pulse, then adjusts for coherent integration over multiple
    pulses.

    Args:
        radar (dict): Dictionary of radar parameters.
        target (dict): Dictionary of target parameters, including 'rcs' and 'range'.
        waveform (dict): Dictionary of waveform parameters, including
                         'time_BW_product'.

    Returns:
        float: The expected SNR as a linear voltage ratio.
    """
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
    #TODO Is skin_snr calculated correctly?
    return np.sqrt(SNR_onepulse / radar["Npulses"])


def add_returns_snr(datacube, waveform, return_list, radar):
    """Adds multiple returns to a datacube, with amplitudes in terms of SNR.

    Iterates through a list of returns (e.g., skin, memory) and adds each
    to the datacube. The amplitude of each return is calculated based on
    the expected SNR. The datacube is modified in place.

    Note: The SNR for memory returns is notional and not physically based.

    Args:
        datacube (np.ndarray): The 2D complex datacube to modify.
        waveform (dict): Dictionary of waveform parameters.
        return_list (list[dict]): A list of dictionaries, where each dictionary
                                  describes a return to be added.
        radar (dict): Dictionary of radar parameters.
    """
    for returnItem in return_list:
        snr_volt_amp = skin_snr_amplitude(radar, returnItem["target"], waveform)
        if returnItem["type"] == "skin":
            add_skin(datacube, waveform, returnItem["target"], radar, snr_volt_amp)
        elif returnItem["type"] == "memory":
            print("Note: Memory return SNR amplitudes are notional!")
            add_memory(datacube, waveform, radar, returnItem, snr_volt_amp)
        else:
            print(f"{returnItem['type']=} not known, no return added.")


def skin_voltage_amplitude(radar, target):
    """Calculates the received voltage amplitude of a skin return.

    Uses the radar range equation to determine the received power from a target
    and converts it to voltage assuming a specific radar load impedance.

    Args:
        radar (dict): Dictionary of radar parameters, icluding 'txPower', 'txGain',
                      'rxGain', 'Noisefactor', 'totalLosses' and 'opTemp'.
        target (dict): Dictionary of target parameters, including 'rcs' and 'range'.

    Returns:
        float: The received voltage amplitude.
    """
    rxPower = signal_range_eqn(
        radar["txPower"],
        radar["txGain"],
        radar["rxGain"],
        target["rcs"],
        c.C / radar["fcar"],
        target["range"],
        radar["totalLosses"],
    )
    print(f"{rxPower=: .2e}")
    return np.sqrt(c.RADAR_LOAD * rxPower)


def memory_voltage_amplitude(platform, radar, target):
    """Calculates the received voltage amplitude of a memory-based EA return.

    Uses a one-way range equation to model the signal received by the radar
    from an EA platform (jammer).

    Args:
        platform (dict): Dictionary of the EA platform's parameters, such as
                         'txPower', 'txGain', 'totalLosses'.
        radar (dict): Dictionary of the radar's parameters, used for 'rxGain'
                      and 'fcar'.
        target (dict): Dictionary of the target's parameters, used for 'range'.

    Returns:
        float: The received voltage amplitude from the EA platform.
    """
    rxMemPower = signal_range_eqn(
        platform["txPower"],
        platform["txGain"],
        radar["rxGain"],
        1,
        c.C / radar["fcar"],  # same as radar if memory
        target["range"] / 2,  # only one-way propagation
        platform["totalLosses"],
    )
    print(f"{rxMemPower=: .2e}")
    return np.sqrt(c.RADAR_LOAD * rxMemPower)


def add_returns(datacube, waveform, return_list, radar):
    """Adds multiple returns to a datacube, with physically-based voltage amplitudes.

    Iterates through a list of returns (e.g., skin, memory) and adds each
    to the datacube. The amplitude of each return is calculated based on the
    radar range equation to find the received voltage. The datacube is
    modified in place.

    Args:
        datacube (np.ndarray): The 2D complex datacube to modify.
        waveform (dict): Dictionary of waveform parameters.
        return_list (list[dict]): A list of dictionaries, where each dictionary
                                  describes a return to be added.
        radar (dict): Dictionary of radar parameters.
    """
    for returnItem in return_list:
        if returnItem["type"] == "skin":
            rx_skin_amp = skin_voltage_amplitude(radar, returnItem["target"])
            add_skin(datacube, waveform, returnItem["target"], radar, rx_skin_amp)
        elif returnItem["type"] == "memory":
            # radar below should should be replaced by EW system
            rx_mem_amp = memory_voltage_amplitude(
                returnItem["platform"], radar, returnItem["target"]
            )
            add_memory(datacube, waveform, radar, returnItem, rx_mem_amp)
        else:
            print(f"{returnItem['type']=} not known, no return added.")


def process_waveform_dict(waveform: dict, radar: dict):
    """Generates waveform samples and computes parameters based on a dictionary.

    This function acts as a factory, creating the actual pulse waveform array
    and calculating key properties like time-bandwidth product and pulse width
    based on the 'type' specified in the waveform dictionary.

    The input `waveform` dictionary is updated in-place with the following keys:
    'pulse', 'time_BW_product', 'pulse_width'.

    Args:
        waveform (dict): Dictionary defining the waveform type and its specific
                         parameters (e.g., 'bw', 'nchips', 'T').
        radar (dict): Dictionary of radar parameters, including 'sampRate'.

    Raises:
        Exception: If the waveform 'type' is not recognized.
    """
    if waveform["type"] == "uncoded":
        _, pulse_wvf = wvf.uncoded_pulse(radar["sampRate"], waveform["bw"])
        waveform["pulse"] = pulse_wvf
        waveform["time_BW_product"] = 1
        waveform["pulse_width"] = 1 / waveform["bw"]

    elif waveform["type"] == "barker":
        _, pulse_wvf = wvf.barker_coded_pulse(
            radar["sampRate"], waveform["bw"], waveform["nchips"]
        )
        waveform["pulse"] = pulse_wvf
        waveform["time_BW_product"] = waveform["nchips"]
        waveform["pulse_width"] = 1 / waveform["bw"] * waveform["nchips"]

    elif waveform["type"] == "random":
        _, pulse_wvf = wvf.random_coded_pulse(
            radar["sampRate"], waveform["bw"], waveform["nchips"]
        )
        waveform["pulse"] = pulse_wvf
        waveform["time_BW_product"] = waveform["nchips"]
        waveform["pulse_width"] = 1 / waveform["bw"] * waveform["nchips"]

    elif waveform["type"] == "lfm":
        _, pulse_wvf = wvf.lfm_pulse(
            radar["sampRate"], waveform["bw"], waveform["T"], waveform["chirpUpDown"]
        )
        waveform["pulse"] = pulse_wvf
        waveform["time_BW_product"] = waveform["bw"] * waveform["T"]
        waveform["pulse_width"] = waveform["T"]

    else:
        raise Exception(f"waveform type {waveform['type']} not found.")
