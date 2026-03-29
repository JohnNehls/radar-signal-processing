import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Dict, List, Any

from . import constants as c
from . import waveform as wvf
from .waveform_helpers import add_waveform_at_index
from .utilities import phase_negpi_pospi
from .range_equation import snr_range_eqn, signal_range_eqn
from . import vbm
from .pulse_doppler_radar import Radar


def add_skin(
    datacube: np.ndarray,
    wvf: Dict,
    tgt_info: Dict,
    radar: Radar,
    return_magnitude: float,
) -> None:
    """Adds a direct radar reflection (skin return) from a target to the datacube.

    This function simulates the signal received by the radar after it reflects
    off a target. The function calculates the time delay and phase shift for each
    pulse and adds the appropriately modified waveform to the datacube.
    The datacube is modified in place.

    Args:
        datacube: 2D complex array to which the return is added.
        wvf: Dictionary containing waveform parameters.
        tgt_info: Dictionary with target information ('range', 'rangeRate', 'sv').
        radar: Radar system parameters.
        return_magnitude: The voltage or SNR amplitude of the return for a single pulse.
    """
    pulse_tx_times = np.arange(radar.Npulses) / radar.PRF

    target_range_per_pulse = tgt_info["range"] + tgt_info["rangeRate"] * pulse_tx_times
    two_way_delays = 2 * target_range_per_pulse / c.C
    pulse_return_times = pulse_tx_times + two_way_delays
    two_way_doppler_phases = -2 * np.pi * radar.fcar * two_way_delays

    # Pulses are timed from their start; compensate with a half pulse-width offset.
    time_pw_offset = wvf["pulse_width"] / 2

    # The range axis is 1-indexed: r_axis[k] = (k+1)*dR, so the injection index
    # must be one less than round(t*fs) to land the MF peak in the correct bin.
    times_of_arrival = pulse_return_times - time_pw_offset
    return_sample_indices = np.round(times_of_arrival * radar.sampRate).astype(int) - 1

    # Due to the time axis being the non-continuous (slow) axis, we must transpose
    flat_datacube = datacube.T.flatten()
    sv = tgt_info.get("sv", 1)

    for i in range(radar.Npulses):
        if return_sample_indices[i] < datacube.size:
            pulse = return_magnitude * wvf["pulse"] * np.exp(1j * two_way_doppler_phases[i]) * sv
            add_waveform_at_index(flat_datacube, pulse, return_sample_indices[i])

    # Reshape the flattened array back to the original datacube shape
    datacube[:] = flat_datacube.reshape(tuple(reversed(datacube.shape))).T


def add_memory(
    datacube: np.ndarray, wvf: Dict, radar: Radar, return_info: Dict, return_magnitude: float
) -> None:
    """Adds a notional memory-based electronic attack (EA) return to the datacube.

    This function simulates a DRFM jammer that records an incoming pulse and
    re-transmits it with modifications to deceive the radar.
    The datacube is modified in place.

    Args:
        datacube: 2D complex array to which the return is added.
        wvf: Dictionary containing waveform parameters.
        radar: Radar system parameters.
        return_info: Dictionary with EA and target information.
        return_magnitude: The voltage or SNR amplitude of the return.
    """
    target = return_info["target"]
    pulse_tx_times = np.arange(radar.Npulses) / radar.PRF

    # Calculate timing and phase for the signal's one-way trip to the target
    target_range_per_pulse = target["range"] + target["rangeRate"] * pulse_tx_times
    one_way_delays = target_range_per_pulse / c.C
    skin_return_times = pulse_tx_times + 2 * one_way_delays
    one_way_propagation_phases = -2 * np.pi * radar.fcar * one_way_delays

    time_pw_offset = wvf["pulse_width"] / 2

    # Doppler frequency shift for range-rate offset
    doppler_freq_offset = 2 * radar.fcar / c.C * return_info.get("rdot_offset", 0)

    # Phase modulation for Velocity Bin Masking (VBM)
    if "rdot_delta" in return_info:
        vbm_noise_function = return_info.get("vbm_noise_function", vbm._lfm_phase)
        slowtime_noise = vbm.slowtime_noise(
            radar.Npulses,
            radar.fcar,
            return_info["rdot_delta"],
            radar.PRF,
            noiseFun=vbm_noise_function,
        )
    else:
        slowtime_noise = np.ones(radar.Npulses)

    # Additional time delay for range offset
    total_delay = return_info.get("delay", 0) + 2 * return_info.get("range_offset", 0) / c.C

    # The range axis is 1-indexed: r_axis[k] = (k+1)*dR, so the injection index
    # must be one less than round(t*fs) to land the MF peak in the correct bin.
    pulse_indices = np.arange(radar.Npulses)
    times_of_arrival = skin_return_times + total_delay - time_pw_offset
    return_sample_indices = np.round(times_of_arrival * radar.sampRate).astype(int) - 1

    # Precompute per-pulse rdot-offset phase shift vector
    rdot_phase = np.exp(-1j * pulse_indices * 2 * np.pi * doppler_freq_offset / radar.PRF)

    stored_pulse = 0
    stored_angle = 0
    sv = target.get("sv", 1)
    flat_datacube = datacube.T.flatten()

    for i in range(radar.Npulses):
        received_pulse = wvf["pulse"] * np.exp(1j * one_way_propagation_phases[i])

        if i == 0:
            stored_pulse = received_pulse
            continue

        if i == 1:
            # Estimate target's Doppler phase shift between pulses
            angle_diff = np.angle(received_pulse) - np.angle(stored_pulse)
            stored_angle = np.mean(phase_negpi_pospi(angle_diff))

        pulse = (
            return_magnitude
            * stored_pulse
            * sv
            * slowtime_noise[i]
            * np.exp(1j * i * stored_angle)
            * rdot_phase[i]
            * np.exp(1j * one_way_propagation_phases[i])
        )

        if return_sample_indices[i] < datacube.size:
            add_waveform_at_index(flat_datacube, pulse, return_sample_indices[i])

    datacube[:] = flat_datacube.reshape(tuple(reversed(datacube.shape))).T


def create_window(
    shape: Tuple[int, int], cheby_atten_db: float = 60.0, plot: bool = False
) -> np.ndarray:
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


def skin_snr_amplitude(radar: Radar, target: Dict, waveform: Dict) -> float:
    """Calculates the required per-pulse voltage amplitude to achieve a target SNR.

    Uses the radar range equation to find the SNR after processing, then works
    backward to determine the necessary per-pulse signal amplitude to inject
    into the simulation datacube.

    Args:
        radar: Radar system parameters.
        target: Dictionary of target parameters.
        waveform: Dictionary of waveform parameters.

    Returns:
        The required per-pulse SNR as a linear voltage ratio.
    """
    # Assumes the range equation provides the total SNR after coherent integration
    # over all pulses in the Coherent Processing Interval (CPI).
    snr_after_integration = snr_range_eqn(
        radar.txPower,
        radar.txGain,
        radar.rxGain,
        target["rcs"],
        c.C / radar.fcar,
        target["range"],
        waveform["bw"],
        radar.noiseFactor,
        radar.totalLosses,
        radar.opTemp,
        waveform["time_BW_product"],
    )

    # To find the required per-pulse amplitude, we first find the per-pulse SNR
    # by dividing by the number of pulses (the coherent integration gain).
    snr_per_pulse = snr_after_integration / radar.Npulses

    # The voltage amplitude for a single pulse is the square root of the per-pulse
    # SNR (power ratio), assuming a normalized noise power of 1.0.
    return np.sqrt(snr_per_pulse)


def add_returns_snr(
    datacube: np.ndarray, waveform: Dict, return_list: List[Dict], radar: Radar
) -> None:
    """Adds multiple returns to a datacube, with amplitudes based on SNR.

    The datacube is modified in place.

    Args:
        datacube: The 2D complex datacube to modify.
        waveform: Dictionary of waveform parameters.
        return_list: A list of dictionaries describing each return.
        radar: Radar system parameters.
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


def skin_voltage_amplitude(radar: Radar, target: Dict) -> float:
    """Calculates the received voltage amplitude of a skin return.

    Args:
        radar: Radar system parameters.
        target: Dictionary of target parameters.

    Returns:
        The received voltage amplitude.
    """
    rx_power = signal_range_eqn(
        radar.txPower,
        radar.txGain,
        radar.rxGain,
        target["rcs"],
        c.C / radar.fcar,
        target["range"],
        radar.totalLosses,
    )
    return np.sqrt(c.RADAR_LOAD * rx_power)


def memory_voltage_amplitude(platform: Dict, radar: Radar, target: Dict) -> float:
    """Calculates the received voltage amplitude of a memory-based EA return.

    Models the one-way communication link from the EA platform to the radar.

    Args:
        platform: Dictionary of the EA platform's parameters.
        radar: Radar system parameters (as receiver).
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
        radar.rxGain,
        equivalent_rcs,
        c.C / radar.fcar,
        range_m,
        platform["totalLosses"],
    )
    return np.sqrt(c.RADAR_LOAD * rx_power)


def add_returns(
    datacube: np.ndarray, waveform: Dict, return_list: List[Dict], radar: Radar
) -> None:
    """Adds multiple returns to a datacube, with physically-based voltage amplitudes.

    The datacube is modified in place.

    Args:
        datacube: The 2D complex datacube to modify.
        waveform: Dictionary of waveform parameters.
        return_list: A list of dictionaries describing each return.
        radar: Radar system parameters.
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


def process_waveform_dict(waveform: Dict[str, Any], radar: Radar) -> None:
    """Generates waveform samples and computes parameters based on a dictionary.

    This function acts as a factory, creating the pulse array and calculating
    key properties based on the 'type' specified in the waveform dictionary.
    The input `waveform` dictionary is updated in-place.

    Args:
        waveform: Dictionary defining the waveform type and its parameters.
                  It will be updated with 'pulse', 'time_BW_product', 'pulse_width'.
        radar: Radar system parameters.

    Raises:
        ValueError: If the waveform 'type' is not recognized.
    """
    samp_rate = radar.sampRate
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
