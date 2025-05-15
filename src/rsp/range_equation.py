from . import constants as c


def signal_range_eqn(Pt, Gt, Gr, sigma, wavelength, R, L):
    """
    Signal power in Watts.
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        R (float) : Range [m]
        L (float) : Loss [unitless]
    Return:
        power : float [W]
    """
    return (Pt * Gt * Gr * sigma * wavelength**2) / (((4 * c.PI) ** 3) * (R**4) * L)


def noise_power(B, F, T):
    """
    Noise power in Watts.
    Args:
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]
        T (float) : Temperature of the system [Kelvin]
    Return:
        power : float [W]
    """
    return c.K_BOLTZ * T * B * F


def snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T):
    """
    Single-pulse SNR for an uncoded pulse. Gt, Gr, F, and L can be either unitless or dB, but all must be te same units.
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        R (float) : Range [m]
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]    
        L (float) : Loss [unitless]    
        T (float) : Temperature of the system [Kelvin]
    Return:
        SNR : float [unitless]
    """
    return (Pt * Gt * Gr * sigma * wavelength**2) / (
        ((4 * c.PI) ** 3) * (R**4) * c.K_BOLTZ * T * B * F * L
    )


def snr_range_eqn(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, time_bandwidth_prod):
    """
    Single-pulse SNR for a general pulse. Gt, Gr, F, and L can be either unitless or dB, but all must be te same units.
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        R (float) : Range [m]
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]
        L (float) : Loss [unitless]
        T (float) : Temperature of the system [Kelvin]
        time_bandwidth_prod (float) : time-bandwidth product [unitless]
    Return:
        SNR : float [unitless]
    """
    return (
        snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T) * time_bandwidth_prod
    )


def snr_range_eqn_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, time_bandwidth_prod):
    """SNR after coherent processing of a number of pulses. Gt, Gr, F, and L can be either unitless or dB, but all must be te same units.
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        R (float) : Range [m]
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]
        L (float) : Loss [unitless]
        T (float) : Temperature of the system [Kelvin]
        n_p (float) : number of pulses coherently processed [unitless]
        time_bandwidth_prod (float) : time-bandwidth product [unitless]
    Return:
        SNR : float [unitless]
    """
    singlePulse_snr = snr_range_eqn(
        Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, time_bandwidth_prod
    )
    return singlePulse_snr * n_p


def snr_range_eqn_bpsk_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, n_c):
    """
    SNR after coherent processing a number of binary phase shift keying pulses. Gt, Gr, F, and L can be either unitless or dB, but all must be te same units.
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        R (float) : Range [m]
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]
        L (float) : Loss [unitless]
        T (float) : Temperature of the system [Kelvin]
        n_p (float) : number of pulses coherently processed [unitless]
        n_c (float) : number of binary chips [unitless]
    Return:
        SNR : float [unitless]
    """
    return snr_range_eqn_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, n_c)


def snr_rangeEquation_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength, R, F, L, T, Tcpi, tau_df):
    """SNR of range equation with coherent processing in duty factor form

    SNR after coherent processing a number of uncodedpulses in duty factor form. Gt, Gr, F, and L can be either unitless or dB, but all must be te same units.
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        R (float) : Range [m]
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]
        L (float) : Loss [unitless]
        T (float) : Temperature of the system [Kelvin]
        Tcpi (float ): Total time of coherent processing interval (CPI) [s]
        tau_df (float) : duty factor, in [0,1] [unitless]
    Return:
        SNR : float [unitless]
    """
    assert tau_df <= 1 and tau_df >= 0, "duty factor must be in [0,1]."

    singlePulse_snr = snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, 1, F, L, T)
    return singlePulse_snr * Tcpi * tau_df


def min_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T):
    """
    Minimum detectable range for a single uncoded pulse. Gt, Gr, F, and L can be either unitless or dB, but all must be te same units.
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        SNR_thresh (float) : SNR threshold
        R (float) : Range [m]
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]
        L (float) : Loss [unitless]
        T (float) : Temperature of the system [Kelvin]
    Return:
        range : float [m]
    """
    return (
        (Pt * Gt * Gr * sigma * wavelength**2)
        / (((4 * c.PI) ** 3) * (SNR_thresh) * c.K_BOLTZ * T * B * F * L)
    ) ** (1 / 4)


def min_target_detection_range_bpsk_cp(
    Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T, n_p, n_c
):
    """
    Minimum detectable range for coherntly processed BPSK pulses. Gt, Gr, F, and L can be either unitless or dB, but all must be te same units.
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        SNR_thresh (float) : SNR threshold
        R (float) : Range [m]
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]
        L (float) : Loss [unitless]
        T (float) : Temperature of the system [Kelvin]
        n_p (float) : number of pulses coherently processed [unitless]
        n_c (float) : number of binary chips [unitless]
    Return:
        range : float [m]
    """
    onePulse = min_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T)
    return onePulse * (n_p * n_c) ** (1 / 4)


def min_target_detection_range_dutyfactor_cp(
    Pt, Gt, Gr, sigma, wavelength, SNR_thresh, F, L, T, Tcpi, tau_df
):
    """
    Minimum detectable range for coherntly processed uncoded pulses in duty factor form. Gt, Gr, F, and L can be either unitless or dB, but all must be te same units.
    Args:
        Pt (float) : Transmit power [W]
        Gt (float) : Transmit gain [unitless]
        Gr (float) : Recieve gain [unitless]
        sigma (float) : Radar cross section [m^2]
        wavelength (float) : Wavelength [Hz]
        SNR_thresh (float) : SNR threshold
        R (float) : Range [m]
        B (float) : Bandwidth of the reciever [Hz]
        F (float) : Noise factor of the reciever [unitless]
        L (float) : Loss [unitless]
        T (float) : Temperature of the system [Kelvin]
        Tcpi (float ): Total time of coherent processing interval (CPI) [s]
        tau_df (float) : duty factor, in [0,1] [unitless]
    Return:
        range : float [m]
    """
    onePulse = min_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, 1, F, L, T)
    return onePulse * (Tcpi * tau_df) ** (1 / 4)
