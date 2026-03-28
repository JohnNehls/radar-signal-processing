import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Dict, List, Any

from . import constants as c
from . import waveform as wvf
from .waveform_helpers import add_waveform_at_index
from .utilities import phase_negpi_pospi, zero_to_smallest_float
from .range_equation import snr_range_eqn, signal_range_eqn
from . import vbm


def plot_rtm(r_axis: np.ndarray, data: np.ndarray, title: str) -> None:
    """Plots the magnitude and phase of a range-time matrix (RTM).

    The RTM shows radar data before Doppler processing, with range on one
    axis and pulse number (slow-time) on the other.

    Args:
        r_axis: 1D array of range values in meters.
        data: 2D complex array representing the RTM, with shape
              (num_range_bins, num_pulses).
        title: The title for the plot.
    """
    pulses = range(data.shape[1])
    fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    mag_plot = ax_mag.pcolormesh(pulses, r_axis * 1e-3, np.abs(data))
    ax_mag.set_xlabel("Pulse Number")
    ax_mag.set_ylabel("Range [km]")
    ax_mag.set_title("Magnitude")
    fig.colorbar(mag_plot, ax=ax_mag)

    phase_plot = ax_phase.pcolormesh(pulses, r_axis * 1e-3, np.angle(data))
    ax_phase.set_xlabel("Pulse Number")
    ax_phase.set_ylabel("Range [km]")
    ax_phase.set_title("Phase")
    fig.colorbar(phase_plot, ax=ax_phase)

    fig.tight_layout()
    plt.show()


def plot_rdm(
    rdot_axis: np.ndarray,
    r_axis: np.ndarray,
    data: np.ndarray,
    title: str,
    cbar_min: float = -100,
    volt_to_dbm: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a range-Doppler matrix (RDM).

    The RDM shows radar data after pulse compression and Doppler processing.

    Args:
        rdot_axis: 1D array of range-rate values in m/s.
        r_axis: 1D array of range values in meters.
        data: 2D complex array representing the RDM.
        title: The title for the plot.
        cbar_min: The minimum value for the color bar. Defaults to -100.
        volt_to_dbm: If True, converts data from voltage to dBm for plotting.
                       If False, plots power in Watts. Defaults to True.

    Returns:
        The figure and axes objects of the plot.
    """
    magnitude_data = np.abs(data)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")

    if volt_to_dbm:
        zero_to_smallest_float(magnitude_data)
        # P_dBm = 10*log10(P_W / 1mW) = 10*log10((V^2/R) / 1e-3)
        plot_data = 20 * np.log10(magnitude_data / np.sqrt(1e-3 * c.RADAR_LOAD))
        cbar_label = "Power [dBm]"
    else:
        # P_W = V^2 / R
        plot_data = magnitude_data**2 / c.RADAR_LOAD
        cbar_label = "Power [W]"

    mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data)
    mesh.set_clim(cbar_min, plot_data.max())
    cbar = fig.colorbar(mesh)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    return fig, ax


def plot_rdm_snr(
    rdot_axis: np.ndarray,
    r_axis: np.ndarray,
    data: np.ndarray,
    title: str,
    cbar_min: float = 0,
    volt_ratio_to_db: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a range-Doppler matrix in terms of Signal-to-Noise Ratio (SNR).

    Args:
        rdot_axis: 1D array of range-rate values in m/s.
        r_axis: 1D array of range values in meters.
        data: 2D array representing the RDM with amplitudes as a linear SNR
              voltage ratio (i.e., S_voltage / N_voltage).
        title: The title for the plot.
        cbar_min: The minimum value for the color bar. Defaults to 0.
        volt_ratio_to_db: If True, converts the SNR voltage ratio to dB.
                            Defaults to True.

    Returns:
        The figure and axes objects of the plot.
    """
    snr_voltage_ratio = np.abs(data)
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")

    if volt_ratio_to_db:
        zero_to_smallest_float(snr_voltage_ratio)
        plot_data = 20 * np.log10(snr_voltage_ratio)
        cbar_label = "SNR [dB]"
    else:
        plot_data = snr_voltage_ratio
        cbar_label = "SNR (Voltage Ratio)"

    mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data)
    mesh.set_clim(cbar_min, plot_data.max())
    cbar = fig.colorbar(mesh)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    return fig, ax


def add_skin(
    datacube: np.ndarray,
    wvf: Dict,
    tgt_info: Dict,
    radar: Dict,
    return_magnitude: float,
):
    """Adds a direct radar reflection (skin return) from a target to the datacube.

    This function simulates the signal received by the radar after it reflects
    off a target. The function calculates the time delay and phase shift for each
    pulse and adds the appropriately modified waveform to the datacube.
    The datacube is modified in place.

    Args:
        datacube: 2D complex array to which the return is added.
        wvf: Dictionary containing waveform parameters.
        tgt_info: Dictionary with target information ('range', 'rangeRate', 'sv').
        radar: Dictionary with radar parameters.
        return_magnitude: The voltage or SNR amplitude of the return for a single pulse.
    """
    full_time_axis = np.arange(datacube.size) / radar["sampRate"]
    pulse_tx_times = np.arange(radar["Npulses"]) / radar["PRF"]

    target_range_per_pulse = tgt_info["range"] + tgt_info["rangeRate"] * pulse_tx_times
    two_way_delays = 2 * target_range_per_pulse / c.C
    pulse_return_times = pulse_tx_times + two_way_delays
    two_way_doppler_phases = -2 * np.pi * radar["fcar"] * two_way_delays

    # Pulses are timed from their start; compensate with a half pulse-width offset.
    time_pw_offset = wvf["pulse_width"] / 2

    # Due to the time axis being the non-continuous (slow) axis, we must transpose
    flat_datacube = datacube.T.flatten()

    for i in range(radar["Npulses"]):
        # Find the sample index corresponding to the pulse's return time.
        # Note: The -1 adjustment is an empirical correction to align the return
        # in the correct range bin for the simulation framework.
        time_of_arrival = pulse_return_times[i] - time_pw_offset
        return_sample_index = np.argmin(np.abs(full_time_axis - time_of_arrival))
        if return_sample_index > 0:
            return_sample_index -= 1

        if return_sample_index < datacube.size:
            phase_shifted_pulse = wvf["pulse"] * np.exp(1j * two_way_doppler_phases[i])
            pulse = return_magnitude * phase_shifted_pulse
            pulse *= tgt_info.get("sv", 1)  # Apply steering vector if available

            add_waveform_at_index(flat_datacube, pulse, return_sample_index)

    # Reshape the flattened array back to the original datacube shape
    datacube[:] = flat_datacube.reshape(tuple(reversed(datacube.shape))).T


def add_memory(
    datacube: np.ndarray, wvf: Dict, radar: Dict, return_info: Dict, return_magnitude: float
):
    """Adds a notional memory-based electronic attack (EA) return to the datacube.

    This function simulates a DRFM jammer that records an incoming pulse and
    re-transmits it with modifications to deceive the radar.
    The datacube is modified in place.

    Args:
        datacube: 2D complex array to which the return is added.
        wvf: Dictionary containing waveform parameters.
        radar: Dictionary with radar parameters.
        return_info: Dictionary with EA and target information.
        return_magnitude: The voltage or SNR amplitude of the return.
    """
    target = return_info["target"]
    full_time_axis = np.arange(datacube.size) / radar["sampRate"]
    pulse_tx_times = np.arange(radar["Npulses"]) / radar["PRF"]

    # Calculate timing and phase for the signal's one-way trip to the target
    target_range_per_pulse = target["range"] + target["rangeRate"] * pulse_tx_times
    one_way_delays = target_range_per_pulse / c.C
    skin_return_times = pulse_tx_times + 2 * one_way_delays
    one_way_propagation_phases = -2 * np.pi * radar["fcar"] * one_way_delays

    time_pw_offset = wvf["pulse_width"] / 2

    # Doppler frequency shift for range-rate offset
    doppler_freq_offset = 2 * radar["fcar"] / c.C * return_info.get("rdot_offset", 0)

    # Phase modulation for Velocity Bin Masking (VBM)
    if "rdot_delta" in return_info:
        vbm_noise_function = return_info.get("vbm_noise_function", vbm._lfm_phase)
        slowtime_noise = vbm.slowtime_noise(
            radar["Npulses"],
            radar["fcar"],
            return_info["rdot_delta"],
            radar["PRF"],
            noiseFun=vbm_noise_function,
        )
    else:
        slowtime_noise = np.ones(radar["Npulses"])

    # Additional time delay for range offset
    total_delay = return_info.get("delay", 0) + 2 * return_info.get("range_offset", 0) / c.C

    stored_pulse = 0
    stored_angle = 0
    flat_datacube = datacube.T.flatten()

    for i in range(radar["Npulses"]):
        received_pulse = wvf["pulse"] * np.exp(1j * one_way_propagation_phases[i])

        if i == 0:
            stored_pulse = received_pulse
            continue

        if i == 1:
            # Estimate target's Doppler phase shift between pulses
            angle_diff = np.angle(received_pulse) - np.angle(stored_pulse)
            stored_angle = np.mean(phase_negpi_pospi(angle_diff))

        # Start with the stored pulse waveform
        pulse = return_magnitude * stored_pulse
        pulse *= target.get("sv", 1)
        pulse *= slowtime_noise[i]  # Apply VBM phase
        pulse *= np.exp(1j * i * stored_angle)  # Apply target's Doppler
        pulse *= np.exp(-1j * i * 2 * np.pi * doppler_freq_offset / radar["PRF"])  # Apply rdot offset
        pulse *= np.exp(1j * one_way_propagation_phases[i])  # Apply 1-way phase back to radar

        time_of_arrival = skin_return_times[i] + total_delay - time_pw_offset
        return_sample_index = np.argmin(np.abs(full_time_axis - time_of_arrival))
        if return_sample_index > 0:
            return_sample_index -= 1

        if return_sample_index < datacube.size:
            add_waveform_at_index(flat_datacube, pulse, return_sample_index)

    datacube[:] = flat_datacube.reshape(tuple(reversed(datacube.shape))).T

def create_window(shape: Tuple[int, int],
                  cheby_atten_db: float = 60.0,
                  plot: bool = False) -> np.ndarray:
    """Creates a 2D Dolph-Chebyshev window for sidelobe reduction.

    The window is applied along the slow-time (pulse) dimension.

    Args:
        shape: Desired shape of the output window (num_range_bins, num_pulses).
        cheby_atten_db: Sidelobe attenuation in dB for the Chebyshev window.
        plot: If True, displays a plot of the generated window.

    Returns:
        The 2D window matrix of shape `shape`.
    """
    assert len(shape) == 2, "Shape must be a 2-element tuple."
    num_range_bins, num_pulses = shape

    cheby_win_1d = signal.windows.chebwin(num_pulses, cheby_atten_db)
    normalized_win = cheby_win_1d / np.mean(cheby_win_1d)
    window_matrix = np.tile(normalized_win, (num_range_bins, 1))

    if plot:
        plt.figure()
        plt.title(f"Dolph-Chebyshev Window ({cheby_atten_db} dB)")
        plt.imshow(window_matrix)
        plt.xlabel("Slow Time (Pulses)")
        plt.ylabel("Fast Time (Range Bins)")
        plt.colorbar(label="Amplitude")
        plt.show()

    return window_matrix


def skin_snr_amplitude(radar: Dict, target: Dict, waveform: Dict) -> float:
    """Calculates the required per-pulse voltage amplitude to achieve a target SNR.

    Uses the radar range equation to find the SNR after processing, then works
    backward to determine the necessary per-pulse signal amplitude to inject
    into the simulation datacube.

    Args:
        radar: Dictionary of radar parameters.
        target: Dictionary of target parameters.
        waveform: Dictionary of waveform parameters.

    Returns:
        The required per-pulse SNR as a linear voltage ratio.
    """
    # Assumes the range equation provides the total SNR after coherent integration
    # over all pulses in the Coherent Processing Interval (CPI).
    snr_after_integration = snr_range_eqn(
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

    # To find the required per-pulse amplitude, we first find the per-pulse SNR
    # by dividing by the number of pulses (the coherent integration gain).
    snr_per_pulse = snr_after_integration / radar["Npulses"]

    # The voltage amplitude for a single pulse is the square root of the per-pulse
    # SNR (power ratio), assuming a normalized noise power of 1.0.
    return np.sqrt(snr_per_pulse)

def add_returns_snr(datacube: np.ndarray, waveform: Dict, return_list: List[Dict], radar: Dict):
    """Adds multiple returns to a datacube, with amplitudes based on SNR.

    The datacube is modified in place.

    Args:
        datacube: The 2D complex datacube to modify.
        waveform: Dictionary of waveform parameters.
        return_list: A list of dictionaries describing each return.
        radar: Dictionary of radar parameters.
    """
    for item in return_list:
        if item["type"] == "skin":
            snr_volt_amp = skin_snr_amplitude(radar, item["target"], waveform)
            add_skin(datacube, waveform, item["target"], radar, snr_volt_amp)
        elif item["type"] == "memory":
            print("Note: Using notional SNR for memory return amplitude.")
            snr_volt_amp = skin_snr_amplitude(radar, item["target"], waveform)
            add_memory(datacube, waveform, radar, item, snr_volt_amp)
        else:
            print(f"Return type '{item['type']}' not recognized. No return added.")


def skin_voltage_amplitude(radar: Dict, target: Dict) -> float:
    """Calculates the received voltage amplitude of a skin return.

    Args:
        radar: Dictionary of radar parameters.
        target: Dictionary of target parameters.

    Returns:
        The received voltage amplitude.
    """
    rx_power = signal_range_eqn(
        radar["txPower"],
        radar["txGain"],
        radar["rxGain"],
        target["rcs"],
        c.C / radar["fcar"],
        target["range"],
        radar["totalLosses"],
    )
    return np.sqrt(c.RADAR_LOAD * rx_power)


def memory_voltage_amplitude(platform: Dict, radar: Dict, target: Dict) -> float:
    """Calculates the received voltage amplitude of a memory-based EA return.

    Models the one-way communication link from the EA platform to the radar.

    Args:
        platform: Dictionary of the EA platform's parameters.
        radar: Dictionary of the radar's parameters (as receiver).
        target: Dictionary of the target's parameters (for range).

    Returns:
        The received voltage amplitude from the EA platform.
    """
    # To model a one-way link using the two-way radar range equation,
    # we can use an effective RCS that cancels the extra 1/(4*pi*R^2) term.
    # The effective RCS is sigma = 4 * pi * R^2.
    range_m = target["range"]
    equivalent_rcs = 4 * np.pi * range_m**2

    rx_power = signal_range_eqn(
        platform["txPower"],
        platform["txGain"],
        radar["rxGain"],
        equivalent_rcs,
        c.C / radar["fcar"],
        range_m,
        platform["totalLosses"],
    )
    return np.sqrt(c.RADAR_LOAD * rx_power)


def add_returns(datacube: np.ndarray, waveform: Dict, return_list: List[Dict], radar: Dict):
    """Adds multiple returns to a datacube, with physically-based voltage amplitudes.

    The datacube is modified in place.

    Args:
        datacube: The 2D complex datacube to modify.
        waveform: Dictionary of waveform parameters.
        return_list: A list of dictionaries describing each return.
        radar: Dictionary of radar parameters.
    """
    for item in return_list:
        if item["type"] == "skin":
            rx_skin_amp = skin_voltage_amplitude(radar, item["target"])
            add_skin(datacube, waveform, item["target"], radar, rx_skin_amp)
        elif item["type"] == "memory":
            rx_mem_amp = memory_voltage_amplitude(item["platform"], radar, item["target"])
            add_memory(datacube, waveform, radar, item, rx_mem_amp)
        else:
            print(f"Return type '{item['type']}' not recognized. No return added.")


def process_waveform_dict(waveform: Dict[str, Any], radar: Dict[str, Any]):
    """Generates waveform samples and computes parameters based on a dictionary.

    This function acts as a factory, creating the pulse array and calculating
    key properties based on the 'type' specified in the waveform dictionary.
    The input `waveform` dictionary is updated in-place.

    Args:
        waveform: Dictionary defining the waveform type and its parameters.
                  It will be updated with 'pulse', 'time_BW_product', 'pulse_width'.
        radar: Dictionary of radar parameters.

    Raises:
        ValueError: If the waveform 'type' is not recognized.
    """
    samp_rate = radar["sampRate"]
    wvf_type = waveform["type"]

    if wvf_type == "uncoded":
        _, pulse_wvf = wvf.uncoded_pulse(samp_rate, waveform["bw"])
        waveform["pulse"] = pulse_wvf
        waveform["time_BW_product"] = 1
        waveform["pulse_width"] = 1 / waveform["bw"]

    elif wvf_type == "barker":
        _, pulse_wvf = wvf.barker_coded_pulse(samp_rate, waveform["bw"], waveform["nchips"])
        waveform["pulse"] = pulse_wvf
        waveform["time_BW_product"] = waveform["nchips"]
        waveform["pulse_width"] = waveform["nchips"] / waveform["bw"]

    elif wvf_type == "random":
        _, pulse_wvf = wvf.random_coded_pulse(samp_rate, waveform["bw"], waveform["nchips"])
        waveform["pulse"] = pulse_wvf
        waveform["time_BW_product"] = waveform["nchips"]
        waveform["pulse_width"] = waveform["nchips"] / waveform["bw"]

    elif wvf_type == "lfm":
        _, pulse_wvf = wvf.lfm_pulse(
            samp_rate, waveform["bw"], waveform["T"], waveform["chirpUpDown"]
        )
        waveform["pulse"] = pulse_wvf
        waveform["time_BW_product"] = waveform["bw"] * waveform["T"]
        waveform["pulse_width"] = waveform["T"]

    else:
        raise ValueError(f"Waveform type '{wvf_type}' not recognized.")
