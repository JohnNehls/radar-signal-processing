import numpy as np
from scipy import fft
from . import constants as c
from .range_equation import snr_range_eqn, snr_range_eqn_cp


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
