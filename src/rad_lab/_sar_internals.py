"""SAR raw-data generation and azimuth compression internals.

Provides the low-level functions that populate a SAR datacube with point-target
returns and focus the data in the cross-range (azimuth) dimension.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.linalg import norm
from scipy import fft

from . import constants as c
from .geometry import slant_range
from ._rdm_internals import _propagation_phase, _return_sample_indices, _inject_pulses
from .waveform import WaveformSample
from .sar_radar import SarRadar, SarTarget


def _beam_weights(
    platform_positions: np.ndarray,
    target_position: list[float],
    scene_center: list[float] | np.ndarray,
    beamwidth: float,
    pattern: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Computes per-pulse two-way antenna gain weights for a steered beam.

    For each pulse the antenna boresight points from the platform toward
    ``scene_center``.  The off-boresight angle to ``target_position`` is
    computed, and a two-way Gaussian beam pattern is applied.

    Args:
        platform_positions: Platform positions ``(n_pulses, 3)`` [m].
        target_position: Target ``[x, y, z]`` [m].
        scene_center: Boresight reference point(s).  Shape ``(3,)`` for a
            fixed point (spotlight) or ``(n_pulses, 3)`` for per-pulse
            broadside points (stripmap).
        beamwidth: One-way 3-dB beamwidth [rad].
        pattern: Optional callable that maps an array of off-boresight
            angles [rad] to two-way amplitude weights.  Defaults to a
            Gaussian: ``exp(-4 ln2 (θ / beamwidth)²)``.

    Returns:
        Per-pulse amplitude weights with shape ``(n_pulses,)``.
    """
    sc = np.asarray(scene_center)
    tp = np.asarray(target_position)

    # Vectors from each platform position to scene centre and to target
    to_scene = sc - platform_positions  # (n_pulses, 3)
    to_target = tp - platform_positions  # (n_pulses, 3)

    # Norms
    d_scene = norm(to_scene, axis=1)
    d_target = norm(to_target, axis=1)

    # Cosine of the angle between the two direction vectors
    cos_theta = np.sum(to_scene * to_target, axis=1) / (d_scene * d_target)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if pattern is not None:
        return pattern(theta)

    # Default: two-way Gaussian (-3 dB at theta = beamwidth/2, squared for two-way)
    return np.exp(-4 * np.log(2) * (theta / beamwidth) ** 2)


def add_sar_returns(
    datacube: np.ndarray,
    waveform: WaveformSample,
    sar_radar: SarRadar,
    target_list: list[SarTarget],
    platform_positions: np.ndarray,
    beam_weights_fn: Callable[[list[float]], np.ndarray] | None = None,
) -> None:
    """Populates a datacube with SAR point-target returns.

    For each pulse position and each target the function computes the slant
    range, two-way propagation delay and carrier phase, then injects the
    waveform at the corresponding range bin.  The datacube is modified
    in place.

    In spotlight mode, ``beam_weights_fn`` supplies per-pulse amplitude
    weights that model the steered antenna beam pattern.

    Args:
        datacube: 2-D complex array of shape ``(n_range_bins, n_pulses)``.
        waveform: Waveform containing the discrete pulse samples.
        sar_radar: SAR system parameters.
        target_list: List of :class:`SarTarget` point scatterers.
        platform_positions: Platform positions ``(n_pulses, 3)`` [m].
        beam_weights_fn: Optional callable that accepts a target position
            ``[x, y, z]`` and returns per-pulse amplitude weights
            ``(n_pulses,)``.  Used for spotlight beam-pattern weighting.
    """
    n_pulses = datacube.shape[1]
    pulse_tx_times = np.arange(n_pulses) / sar_radar.prf

    for target in target_list:
        # Slant range from every aperture position to this target [m]
        ranges = slant_range(platform_positions, target.position)

        # Two-way propagation delay and carrier phase per pulse
        two_way_delays = 2 * ranges / c.C
        two_way_phases = _propagation_phase(two_way_delays, sar_radar.fcar)

        # Absolute return times (pulse tx time + two-way delay) are needed so
        # that each pulse's waveform lands in the correct section of the flat array.
        return_times = pulse_tx_times + two_way_delays
        sample_indices = _return_sample_indices(return_times, waveform, sar_radar.sample_rate)

        # Amplitude: sqrt(RCS), optionally weighted by beam pattern
        amplitude: float | np.ndarray = np.sqrt(target.rcs)
        if beam_weights_fn is not None:
            amplitude = amplitude * beam_weights_fn(target.position)

        _inject_pulses(
            datacube,
            waveform.pulse_sample,
            sample_indices,
            two_way_phases,
            amplitude=amplitude,
        )


def azimuth_matched_filter(
    datacube: np.ndarray,
    sar_radar: SarRadar,
    range_axis: np.ndarray,
) -> np.ndarray:
    """Focuses a range-compressed datacube in the azimuth (cross-range) dimension.

    For each range bin the function builds a reference phase history based on
    the exact hyperbolic slant-range variation across the synthetic aperture,
    then correlates it with the data via FFT convolution along the slow-time
    axis.

    Args:
        datacube: 2-D complex array ``(n_range_bins, n_pulses)``, already
            range-compressed.
        sar_radar: SAR system parameters.
        range_axis: 1-D range axis [m] with length ``n_range_bins``.

    Returns:
        Cross-range (azimuth) axis [m] with length ``n_pulses``.
    """
    n_range_bins, n_pulses = datacube.shape

    # Along-track positions of each aperture sample, centred at 0
    along_track = (np.arange(n_pulses) - n_pulses / 2) * sar_radar.pulse_spacing

    for k in range(n_range_bins):
        R0 = range_axis[k]

        # Exact hyperbolic range history for a target at broadside range R0
        R_history = np.sqrt(R0**2 + along_track**2)

        # Reference signal: the phase a broadside target would produce
        h_ref = np.exp(-1j * 4 * np.pi / sar_radar.wavelength * R_history)

        # Matched filter = correlation via FFT: ifft(FFT(data) * conj(FFT(h_ref)))
        datacube[k, :] = fft.fftshift(fft.ifft(fft.fft(datacube[k, :]) * np.conj(fft.fft(h_ref))))

    # Cross-range axis maps directly to along-track position
    cross_range_axis = along_track.copy()

    return cross_range_axis
