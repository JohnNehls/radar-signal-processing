"""SAR raw-data generation and azimuth compression internals.

Provides the low-level functions that populate a SAR datacube with point-target
returns and focus the data in the cross-range (azimuth) dimension.
"""

import numpy as np
from scipy import fft

from . import constants as c
from .geometry import slant_range
from .waveform_helpers import add_waveform_at_index
from .waveform import WaveformSample
from .sar_radar import SarRadar, SarTarget


def _sar_sample_indices(
    two_way_delays: np.ndarray, waveform: WaveformSample, sample_rate: float
) -> np.ndarray:
    """Converts two-way delays to flat datacube sample indices.

    Matches the bin-alignment convention in ``_rdm_internals._return_sample_indices``:
    subtract half the pulse width (pulses are timed from their leading edge) and
    offset by one so the matched-filter peak lands in the correct range bin.

    Args:
        two_way_delays: Two-way propagation delays for each pulse [s].
        waveform: Waveform with ``pulse_width`` attribute [s].
        sample_rate: ADC sampling rate [Hz].

    Returns:
        Integer sample indices into the flattened datacube.
    """
    times_of_arrival = two_way_delays - waveform.pulse_width / 2
    return np.round(times_of_arrival * sample_rate).astype(int) - 1


def add_sar_returns(
    datacube: np.ndarray,
    waveform: WaveformSample,
    sar_radar: SarRadar,
    target_list: list[SarTarget],
    platform_positions: np.ndarray,
) -> None:
    """Populates a datacube with SAR point-target returns.

    For each pulse position and each target the function computes the slant
    range, two-way propagation delay and carrier phase, then injects the
    waveform at the corresponding range bin.  The datacube is modified
    in place.

    Args:
        datacube: 2-D complex array of shape ``(n_range_bins, n_pulses)``.
        waveform: Waveform containing the discrete pulse samples.
        sar_radar: SAR system parameters.
        target_list: List of :class:`SarTarget` point scatterers.
        platform_positions: Platform positions ``(n_pulses, 3)`` [m].
    """
    n_range_bins, n_pulses = datacube.shape

    pulse_tx_times = np.arange(n_pulses) / sar_radar.prf

    for target in target_list:
        # Slant range from every aperture position to this target [m]
        ranges = slant_range(platform_positions, target.position)

        # Two-way propagation delay and carrier phase per pulse
        two_way_delays = 2 * ranges / c.C
        two_way_phases = -2 * np.pi * sar_radar.fcar * two_way_delays

        # Amplitude from RCS (simple sqrt(rcs) scaling; SNR mode not yet supported)
        amplitude = np.sqrt(target.rcs)

        # Absolute return times (pulse tx time + two-way delay) are needed so
        # that each pulse's waveform lands in the correct section of the flat array.
        return_times = pulse_tx_times + two_way_delays
        sample_indices = _sar_sample_indices(return_times, waveform, sar_radar.sample_rate)

        # Inject waveform pulse-by-pulse into the flattened datacube
        flat = datacube.T.flatten()
        for i in range(n_pulses):
            if sample_indices[i] < datacube.size:
                pulse = amplitude * waveform.pulse_sample * np.exp(1j * two_way_phases[i])
                add_waveform_at_index(flat, pulse, sample_indices[i])
        datacube[:] = flat.reshape(n_pulses, n_range_bins).T


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
