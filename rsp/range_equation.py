from . import constants as c


def signal_range_eqn(Pt, Gt, Gr, sigma, wavelength, R, L):
    """Signal power in Watts considering transmit and recieve gain, RCS, wavelength, range, and losses
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        R (float) : Range [m]
        L (float) : Loss [unitless]
    Return:
        (float) signal power [S]
    """
    return (Pt * Gt * Gr * sigma * wavelength**2) / (((4 * c.PI) ** 3) * (R**4) * L)


def noise_power(B, F, T):
    """Single-pulse SNR for uncoded pulse
    Args:
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]
        T (float) : Temperature of the system [Kelvin]
    Return:
        Power of the noise
    """
    return c.K_BOLTZ * T * B * F


def snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T):
    """Single-pulse SNR for uncoded pulse
    Gt, Gr, F, and L can be either unitless or dB, but all must be te same units"""
    return (Pt * Gt * Gr * sigma * wavelength**2) / (
        ((4 * c.PI) ** 3) * (R**4) * c.K_BOLTZ * T * B * F * L
    )


def snr_range_eqn(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, time_bandwidth_prod):
    """Single-pulse SNR
    Gt, Gr, F, and L can be either unitless or dB, but all must be te same units"""
    return snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T) * time_bandwidth_prod


def snr_range_eqn_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, time_bandwidth_prod):
    """ "SNR of range equation with coherent processing (CP)
    Gt, Gr, F, and L can be either unitless or dB, but all must be te same units"""
    singlePulse_snr = snr_range_eqn(
        Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, time_bandwidth_prod
    )
    return singlePulse_snr * n_p


def snr_range_eqn_bpsk_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, n_c):
    """ "SNR of range equation with coherent processing (CP)
    \tn_p := number of pulses
    \tn_c := number of chips
        Gt, Gr, F, and L can be either unitless or dB, but all must be te same units
    """
    return snr_range_eqn_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, n_c)


def snr_rangeEquation_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength, R, F, L, T, Tcpi, tau_df):
    """ "SNR of range equation with coherent processing in duty factor form
    \tTcpi := Total time of coherent processing interval (CPI) in seconds
    \ttau_df := duty factor, in [0,1]
        Gt, Gr, F, and L can be either unitless or dB, but all must be te same units
    """
    singlePulse_snr = snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, 1, F, L, T)
    return singlePulse_snr * Tcpi * tau_df


def min_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T):
    """single pulse minimum detectable range for a SNR_threshold
    Gt, Gr, F, and L can be either unitless or dB, but all must be te same units
    """
    return (
        (Pt * Gt * Gr * sigma * wavelength**2)
        / (((4 * c.PI) ** 3) * (SNR_thresh) * c.K_BOLTZ * T * B * F * L)
    ) ** (1 / 4)


def min_target_detection_range_bpsk_cp(
    Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T, n_p, n_c
):
    """Minimum detectable range for a SNR_threshold for BPSK pulses
    \tn_p := number of pulses
    \tn_c := number of chips
        Gt, Gr, F, and L can be either unitless or dB, but all must be te same units
    """
    onePulse = min_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T)
    return onePulse * (n_p * n_c) ** (1 / 4)


def min_target_detection_range_dutyfactor_cp(
    Pt, Gt, Gr, sigma, wavelength, SNR_thresh, F, L, T, Tcpi, tau_df
):
    """Minimum detectable range for a SNR_threshold for BPSK pulses
    \tTcpi := Total time of coherent processing interval (CPI) in seconds
    \tau_df := duty factor, in [0,1]
        Gt, Gr, F, and L can be either unitless or dB, but all must be te same units
    """
    onePulse = min_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, 1, F, L, T)
    return onePulse * (Tcpi * tau_df) ** (1 / 4)
