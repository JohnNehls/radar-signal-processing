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
    """
    Plot range-time matrix.
    Args:
        r_axis (1D array) : Range axis
        data (2D array) : Range-time matrix.
        title (string) : Plot title
    Return:
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
    """
    Plot range-Doppler matrix.
    Args:
        rdot_axis (1D array) : Range rate axis
        r_axis (1D array) : Range axis
        data (2D array) : Range-time matrix
        title (string) : Plot title
        cbarMin (float) : Color bar minimum (default=-100)
        volt2dbm (bool) : Flag to convert to dBm (default=True)
    Return:
        None
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
    """Plot range-Doppler matrix in SNR.
    Args:
        rdot_axis (1D array) : Range rate axis
        r_axis (1D array) : Range axis
        data (2D array) : Range-time matrix
        title (string) : Plot title
        cbarMin (float) : Color bar minimum (default=0)
        volt2dbm (bool) : Flag to convert to dBm (default=True)
    Return:
        None
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
    """
    Add skin return from target to the datacube in place.
    Args:
        datacube (2D array) : Datacube
        wvf (dict) : Waveform dictionary
        tgtInfo (dict) : Target dictionary
        radar (dict) : Radar dictionary
        return_magnitude (float) : Magitude of the return (voltage or SNR)
    Return:
        None
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
    """
    Add notional memory return to datacube in place.
    Args:
        datacube (2D array) : Datacube
        wvf (dict) : Waveform dictionary
        tgtInfo (dict) : Target dictionary
        radar (dict) : Radar dictionary
        return_magnitude (float) : Magitude of the return (voltage or SNR)
    Return:
        None
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
    """
    Create Dolph-Chebyshev window using Numpy.
    Args:
        inShape (tuple) : Window shape
        plot (bool) : Flag to plot the window
    Return:
        chwin_norm_mat
    """
    # see wondow comparison example for more window examples
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
    """
    Return the expected SNR for a given radar, target, and waveform used.
    Args:
        radar (dict) : Radar properties (txPower, txGain, rxGain, Noisefactor, totalLosses opTemp)
        target (dict) : Target properties (rcs)
        waveform (dict) : Waveform properties (time_BW_product)
    Return:
        SNR (float) : Expected SNR
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
    """
    Add each from the return_list to the SNR datacube in place.
    Args:
        datacube (2D array) : Datacube
        waveform (dict) : Waveform properties
        return_list (list) : list of the returns to be added in the RDM
        radar (dict) : Radar properties
    Return:
        None
    Notes:
        - SNR Memory return amplitude is not physical
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
    """
    Calculate the correct voltage amplitude of a return from a target.
    Args:
        radar (dict) : Radar properties (txPower, txGain, rxGain, Noisefactor, totalLosses opTemp)
        target (dict) : Target properties (rcs)
    Return:
        amplitude (float) : volatage amplitude
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
    """
    Calculate the correct voltage amplitude of a memory return from a platform.
    Args:
        platform (dict) : Platform properties (txPower, txgain, totalLosses)
        radar (dict) : Radar properties (rxGain, fcar)
        target (dict) : Target properties (rcs)
    Return:
        amplitude (float) : volatage amplitude
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
    """
    Add each from the return_list to the datacube in place.
    Args:
        datacube (2D array) : Datacube
        waveform (dict) : Waveform properties
        return_list (list) : list of the returns to be added in the RDM
        radar (dict) : Radar properties
    Return:
        None
    Notes:
        - Check if memory return amplitude is physical
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


## see examples/tests/function_tests/process_waveform.py for simple test of this function
def process_waveform_dict(waveform: dict, radar: dict):
    """
    Fill in waveform dict with "pulse", "time_BW_product", "pulse_width" in place.
    Args:
        waveform (dict) : Waveform properties
        radar (dict) : Radar properties
    Return:
        None
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
