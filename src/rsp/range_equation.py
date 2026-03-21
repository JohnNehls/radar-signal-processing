from . import constants as c


def signal_range_eqn(Pt, Gt, Gr, sigma, wavelength, R, L):
    """
    Calculate the received signal power for a radar system.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        L (float): System losses [unitless]

    Returns:
        float: Received signal power [W]
    """
    return (Pt * Gt * Gr * sigma * wavelength**2) / (((4 * c.PI) ** 3) * (R**4) * L)


def noise_power(B, F, T):
    """
    Calculate the thermal noise power of the receiver.

    Args:
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        T (float): System noise temperature [Kelvin]

    Returns:
        float: Noise power [W]
    """
    return c.K_BOLTZ * T * B * F


def snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T):
    """
    Calculate the single-pulse Signal-to-Noise Ratio (SNR) for an uncoded pulse.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]

    Returns:
        float: Signal-to-Noise Ratio [unitless]
    """
    return (Pt * Gt * Gr * sigma * wavelength**2) / (
        ((4 * c.PI) ** 3) * (R**4) * c.K_BOLTZ * T * B * F * L
    )


def snr_range_eqn(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, time_bandwidth_prod):
    """
    Calculate the single-pulse Signal-to-Noise Ratio (SNR) for a pulse with pulse compression.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        time_bandwidth_prod (float): Pulse compression ratio (time-bandwidth product) [unitless]

    Returns:
        float: Signal-to-Noise Ratio [unitless]
    """
    return (
        snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T) * time_bandwidth_prod
    )


def snr_range_eqn_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, time_bandwidth_prod):
    """
    Calculate the Signal-to-Noise Ratio (SNR) after coherent processing of multiple pulses.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        n_p (float): Number of pulses coherently integrated [unitless]
        time_bandwidth_prod (float): Pulse compression ratio [unitless]

    Returns:
        float: Integrated Signal-to-Noise Ratio [unitless]
    """
    singlePulse_snr = snr_range_eqn(
        Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, time_bandwidth_prod
    )
    return singlePulse_snr * n_p


def snr_range_eqn_bpsk_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, n_c):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for Binary Phase Shift Keying (BPSK) pulses after coherent processing.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        n_p (float): Number of pulses coherently integrated [unitless]
        n_c (float): Number of binary chips per pulse [unitless]

    Returns:
        float: Integrated Signal-to-Noise Ratio [unitless]
    """
    return snr_range_eqn_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, n_c)


def snr_rangeEquation_dutyFactor_pulses(Pt, Gt, Gr, sigma, wavelength, R, F, L, T, Tcpi, tau_df):
    """
    Calculate the Signal-to-Noise Ratio (SNR) using the duty factor and coherent processing interval.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Peak transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        Tcpi (float): Total coherent processing interval (CPI) duration [s]
        tau_df (float): Radar duty factor [0 to 1]

    Returns:
        float: Integrated Signal-to-Noise Ratio [unitless]
    """
    assert 0 <= tau_df <= 1, "duty factor must be in [0,1]."

    singlePulse_snr = snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, 1, F, L, T)
    return singlePulse_snr * Tcpi * tau_df


def min_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T):
    """
    Calculate the maximum detectable range for a single uncoded pulse given an SNR threshold.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        SNR_thresh (float): Minimum SNR required for detection [unitless]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]

    Returns:
        float: Maximum detection range [m]
    """
    return (
        (Pt * Gt * Gr * sigma * wavelength**2)
        / (((4 * c.PI) ** 3) * (SNR_thresh) * c.K_BOLTZ * T * B * F * L)
    ) ** (1 / 4)


def min_target_detection_range_bpsk_cp(
    Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T, n_p, n_c
):
    """
    Calculate the maximum detectable range for coherently processed BPSK pulses.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        SNR_thresh (float): Minimum SNR required for detection [unitless]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        n_p (float): Number of pulses coherently processed [unitless]
        n_c (float): Number of binary chips per pulse [unitless]

    Returns:
        float: Maximum detection range [m]
    """
    onePulse = min_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T)
    return onePulse * (n_p * n_c) ** (1 / 4)


def min_target_detection_range_dutyfactor_cp(
    Pt, Gt, Gr, sigma, wavelength, SNR_thresh, F, L, T, Tcpi, tau_df
):
    """
    Calculate the maximum detectable range for coherently processed pulses using duty factor parameters.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Peak transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        SNR_thresh (float): Minimum SNR required for detection [unitless]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        Tcpi (float): Total coherent processing interval duration [s]
        tau_df (float): Radar duty factor [0 to 1]

    Returns:
        float: Maximum detection range [m]
    """
    onePulse = min_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, 1, F, L, T)
    return onePulse * (Tcpi * tau_df) ** (1 / 4)
