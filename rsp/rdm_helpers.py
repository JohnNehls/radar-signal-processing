import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
from . import constants as c
from .waveform_helpers import add_waveform_at_index
from .utilities import phase_negpi_pospi, zero_to_smallest_float
from .range_equation import snr_range_eqn, snr_range_eqn_cp, signal_range_eqn
from . import vbm


def plot_rtm(r_axis, data, title):
    """Plot range-time matrix"""
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
    """Plot range-Doppler matrix"""

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
    """Plot range-Doppler matrix"""
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


def add_skin(signal_dc, wvf: dict, tgtInfo: dict, radar: dict, SNR_volt):
    """Add skin return to the datacube"""
    # time and range arrays
    time_ar = np.arange(signal_dc.size) * 1 / radar["sampRate"]  # time of all samples in CPI
    t_slow_axis = np.arange(radar["Npulses"]) * 1 / radar["PRF"]  # time when pulses sent

    tgt_range_ar = tgtInfo["range"] + tgtInfo["rangeRate"] * t_slow_axis  # tgt range at pulse send
    twoWay_time_delay_ar = 2 * tgt_range_ar / c.C  # time of travel from radar to tgt and back
    pulse_return_time = t_slow_axis + twoWay_time_delay_ar  # time pulses return to radar
    twoWay_phase_ar = -2 * c.PI * radar["fcar"] * twoWay_time_delay_ar  # Phase added due to

    ## pulses timed from their start not their center, we compensate with pw/2 range offset
    time_pw_offset = wvf["pulse_width"] / 2

    # Due to the time axis being the non-continuous (slow) axis, we most do some transposing
    tmpSignal = signal_dc.T.flatten()

    for i in range(radar["Npulses"]):
        # TODO is this how these should be binned? Should they be interpolated onto grid?
        timeIndex = np.argmin(abs(time_ar - pulse_return_time[i] + time_pw_offset))
        pulse = SNR_volt * wvf["pulse"] * np.exp(1j * twoWay_phase_ar[i])

        if timeIndex < signal_dc.size:  # else pulse is in next CPI
            add_waveform_at_index(tmpSignal, pulse, timeIndex)

    tmpSignal = tmpSignal.reshape(tuple(reversed(signal_dc.shape))).T

    signal_dc[:] = tmpSignal[:]


def add_memory(signal_dc, wvf: dict, tgtInfo: dict, radar: dict, returnInfo, SNR_volt):
    """Add notional memory return to datacube"""
    print("Note: memory return amplitudes are notional")

    # time and range arrays
    time_ar = np.arange(signal_dc.size) * 1 / radar["sampRate"]  # time of all samples in CPI
    t_slow_axis = np.arange(radar["Npulses"]) * 1 / radar["PRF"]  # time when pulses sent

    tgt_range_ar = tgtInfo["range"] + tgtInfo["rangeRate"] * t_slow_axis  # tgt range at pulse send
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
    tmpSignal = signal_dc.T.flatten()

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
        pulse = SNR_volt * stored_pulse

        # add slowtime noise (VBM)
        pulse = pulse * slowtime_noise[i]

        # add stored pulse difference rdot
        pulse = pulse * (np.exp(1j * i * stored_angle))

        # add rdot offset
        pulse = pulse * (np.exp(-1j * i * 2 * c.PI * f_rdot / radar["PRF"]))

        # add 1-way propagation phase back to radar
        pulse = pulse * np.exp(1j * oneWay_phase_ar[i])

        # TODO is this how these should be binned? Should they be interpolated onto grid?
        timeIndex = np.argmin(abs(time_ar - pulse_return_time[i] - delay + time_pw_offset))
        if timeIndex < signal_dc.size:  # else pulse is in next CPI
            add_waveform_at_index(tmpSignal, pulse, timeIndex)

    tmpSignal = tmpSignal.reshape(tuple(reversed(signal_dc.shape))).T

    signal_dc[:] = tmpSignal[:]


def noise_checks(signal_dc, noise_dc, total_dc):
    """Print out some noise checks"""
    print(f"\n5.3.2 noise check: {np.var(fft.fft(noise_dc, axis=1))=: .4f}")
    print("\nnoise check:")
    noise_var = np.var(total_dc, 1)
    print(f"\t{np.mean (noise_var)=: .4f}")
    print(f"\t{np.var(noise_var)=: .4f}")
    print(f"\t{np.mean (20*np.log10(noise_var))=: .4f}")
    print(f"\t{np.var (20*np.log10(noise_var))=: .4f}")
    print("\nSNR test:")
    print(f"\t{20*np.log10(np.max(abs(signal_dc)))=:.2f}")
    print(f"\t{20*np.log10(np.max(abs(noise_dc)))=:.2f}")
    print(f"\t{20*np.log10(np.max(abs(total_dc)))=:.2f}")


def check_expected_snr(radar, target, waveform):
    ## expected
    SNR_expected = snr_range_eqn_cp(
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
        radar["Npulses"],
        waveform["time_BW_product"],
    )
    ## volatge used in return (recalculated)
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

    print("SNR Check:")
    print(f"\t{10*np.log10(SNR_onepulse)=:.2f}")
    print(f"\t{SNR_volt=:.1e}")
    print(f"\t{SNR_expected=:.1e}")
    print(f"\t{10*np.log10(SNR_expected)=:.2f}")


def create_window(inShape: tuple, plot=True):
    """Create windowing function"""
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
    print("Is skin_snr calculated correctly?")
    return np.sqrt(SNR_onepulse / radar["Npulses"])


def add_returns_snr(dc, wvf, target, return_list, radar):
    """Add returns from the return_list to the data cube
    Note: memory return amplitude is not physical"""
    snr_volt_amp = skin_snr_amplitude(radar, target, wvf)
    for returnItem in return_list:
        if returnItem["type"] == "skin":
            add_skin(dc, wvf, target, radar, snr_volt_amp)
        elif returnItem["type"] == "memory":
            print("Note: Memory return SNR amplitudes are notional!")
            add_memory(dc, wvf, target, radar, returnItem, snr_volt_amp)
        else:
            print(f"{returnItem['type']=} not known, no return added.")


def skin_voltage_amplitude(radar, target):
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


def add_returns(signal_dc, waveform, target, return_list, radar):
    """Add returns from the return_list to the data cube
    Note: memory return amplitude is not physical"""
    for returnItem in return_list:
        if returnItem["type"] == "skin":
            rx_skin_amp = skin_voltage_amplitude(radar, target)
            add_skin(signal_dc, waveform, target, radar, rx_skin_amp)
        elif returnItem["type"] == "memory":
            # radar below should should be replaced by EW system
            rx_mem_amp = memory_voltage_amplitude(returnItem["platform"], radar, target)
            add_memory(signal_dc, waveform, target, radar, returnItem, rx_mem_amp)
        else:
            print(f"{returnItem['type']=} not known, no return added.")
