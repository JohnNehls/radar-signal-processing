"""SAR system parameters and target models.

Defines :class:`SarRadar` (stripmap SAR platform and system parameters) and
:class:`SarTarget` (a point scatterer located in 3-D space).
"""

import math
from dataclasses import dataclass, field

from . import constants as c


@dataclass
class SarRadar:
    """Stripmap SAR system parameters.

    All gain and loss fields are linear power ratios (not dB).
    Derived quantities (``n_pulses``, ``wavelength``, ``pulse_spacing``)
    are computed automatically in ``__post_init__``.

    Attributes:
        fcar: Carrier frequency [Hz].
        tx_power: Transmit power [W].
        tx_gain: Transmit antenna gain [linear].
        rx_gain: Receive antenna gain [linear].
        op_temp: Receiver operating temperature [K].
        sample_rate: ADC sampling rate [Hz].
        noise_factor: Receiver noise factor [linear].
        total_losses: Total two-way system losses [linear].
        prf: Pulse repetition frequency [Hz].
        platform_velocity: Platform ground speed along the flight path [m/s].
        aperture_length: Synthetic aperture length [m].
        platform_altitude: Platform altitude above the scene [m].
        n_pulses: Number of pulses in the synthetic aperture, computed as
            ``ceil(aperture_length / pulse_spacing)`` [dimensionless].
        wavelength: Carrier wavelength [m], derived from ``fcar``.
        pulse_spacing: Along-track distance between pulses [m], derived from
            ``platform_velocity / prf``.
    """

    fcar: float
    tx_power: float
    tx_gain: float
    rx_gain: float
    op_temp: float
    sample_rate: float
    noise_factor: float
    total_losses: float
    prf: float
    platform_velocity: float
    aperture_length: float
    platform_altitude: float = 0.0
    n_pulses: int = field(init=False)
    wavelength: float = field(init=False)
    pulse_spacing: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived quantities from the input parameters."""
        self.wavelength = c.C / self.fcar
        self.pulse_spacing = self.platform_velocity / self.prf
        self.n_pulses = int(math.ceil(self.aperture_length / self.pulse_spacing))


@dataclass
class SarTarget:
    """A point scatterer in a SAR scene.

    Targets are defined by their 3-D position in a scene-centred coordinate
    system where *x* is along-track (flight direction), *y* is cross-track
    (range direction), and *z* is altitude.

    Attributes:
        position: 3-D Cartesian coordinates ``[x, y, z]`` [m].
            *x* = along-track, *y* = cross-track, *z* = altitude.
        rcs: Radar cross section [m^2].
    """

    position: list[float]
    rcs: float


def cross_range_resolution(wavelength: float, slant_range: float, aperture_length: float) -> float:
    """Computes the cross-range (azimuth) resolution for a stripmap SAR.

    Resolution improves with a longer aperture and degrades at longer range.

    Args:
        wavelength: Carrier wavelength [m].
        slant_range: Slant range to the target [m].
        aperture_length: Synthetic aperture length [m].

    Returns:
        Cross-range 3-dB resolution [m].
    """
    return wavelength * slant_range / (2 * aperture_length)
