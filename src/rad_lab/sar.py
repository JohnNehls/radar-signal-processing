"""Stripmap SAR image generation and plotting.

Provides the :func:`gen` entry point that simulates a full synthetic aperture —
transmitting pulses along a straight flight path, injecting point-target
returns, range-compressing, and azimuth-focusing to produce a SAR image.
"""

import numpy as np
import matplotlib.pyplot as plt

from . import constants as c
from .rf_datacube import number_range_bins, range_axis, data_cube, matchfilter
from .range_equation import noise_power
from .utilities import zero_to_smallest_float
from ._rdm_internals import create_window
from ._sar_internals import add_sar_returns, azimuth_matched_filter
from .geometry import flight_path
from .sar_radar import SarRadar, SarTarget
from .waveform import WaveformSample


def gen(
    sar_radar: SarRadar,
    waveform: WaveformSample,
    target_list: list[SarTarget],
    seed: int = 0,
    plot: bool = True,
    debug: bool = False,
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a focused stripmap SAR image from point-target returns.

    Simulates a SAR collection over a straight, level flight path.  For each
    aperture position the function computes range and phase to every target,
    injects the waveform, then processes the datacube through range
    compression, azimuth windowing, and azimuth matched-filter focusing.

    Args:
        sar_radar: SAR system parameters.
            See :class:`rad_lab.sar_radar.SarRadar`.
        waveform: WaveformSample created by a factory function
            (e.g. :func:`rad_lab.waveform.lfm_waveform`).
        target_list: List of :class:`~rad_lab.sar_radar.SarTarget` point
            scatterers.
        seed: Random number generator seed for reproducibility.
        plot: If True, plots the focused SAR image.
        debug: If True, plots intermediate processing steps (raw data,
            range-compressed data).
        window: Window function applied along the azimuth dimension before
            focusing.  One of ``"chebyshev"`` (default),
            ``"blackman-harris"``, ``"taylor"``, or ``"none"``.
        window_kwargs: Optional dict forwarded to the window function.
            See :func:`._rdm_internals.create_window`.

    Returns:
        tuple: ``(cross_range_axis, r_axis, focused_dc, signal_dc)``:

            - **cross_range_axis** (*np.ndarray*): 1-D cross-range axis [m].
            - **r_axis** (*np.ndarray*): 1-D slant-range axis [m].
            - **focused_dc** (*np.ndarray*): 2-D focused SAR image (signal + noise).
            - **signal_dc** (*np.ndarray*): 2-D signal-only focused image.
    """
    np.random.seed(seed)

    ########## Waveform setup #####################################################################
    waveform.set_sample(sar_radar.sample_rate)

    ########## Flight path ########################################################################
    platform_positions = flight_path(
        sar_radar.n_pulses, sar_radar.pulse_spacing, sar_radar.platform_altitude
    )

    ########## Create datacube and range axis ######################################################
    n_range_bins = number_range_bins(sar_radar.sample_rate, sar_radar.prf)
    r_axis = range_axis(sar_radar.sample_rate, n_range_bins)

    signal_dc = data_cube(sar_radar.sample_rate, sar_radar.prf, sar_radar.n_pulses)

    ########## Populate with target returns ########################################################
    add_sar_returns(signal_dc, waveform, sar_radar, target_list, platform_positions)

    ########## Add noise ##########################################################################
    rx_noise_volt = np.sqrt(
        c.RADAR_LOAD * noise_power(waveform.bw, sar_radar.noise_factor, sar_radar.op_temp)
    )
    noise_dc = np.random.uniform(low=-1, high=1, size=signal_dc.shape) * rx_noise_volt

    total_dc = signal_dc + noise_dc

    if debug:
        _plot_raw(r_axis, signal_dc, "Raw SAR data (noiseless)")

    # Process both signal-only and total datacubes
    dc_list = [signal_dc, total_dc]

    ########## Range compression ##################################################################
    for dc in dc_list:
        matchfilter(dc, waveform.pulse_sample, pedantic=False)

    if debug:
        _plot_raw(r_axis, signal_dc, "Range-compressed (noiseless)")

    ########## Azimuth windowing ##################################################################
    win_mat = create_window(
        signal_dc.shape, window=window, window_kwargs=window_kwargs, plot=False
    )
    for dc in dc_list:
        dc *= win_mat

    ########## Azimuth compression (focusing) #####################################################
    for dc in dc_list:
        cross_range_axis = azimuth_matched_filter(dc, sar_radar, r_axis)

    ########## Plot ###############################################################################
    if plot or debug:
        plot_sar_image(cross_range_axis, r_axis, total_dc, "Focused SAR Image")

    return cross_range_axis, r_axis, total_dc, signal_dc


def plot_sar_image(
    cross_range_axis: np.ndarray,
    r_axis: np.ndarray,
    data: np.ndarray,
    title: str,
    cbar_min: float = -40,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a focused SAR image in dB.

    Args:
        cross_range_axis: 1-D cross-range axis [m].
        r_axis: 1-D slant-range axis [m] (converted to km for display).
        data: 2-D complex SAR image.
        title: Plot title.
        cbar_min: Minimum colorbar value [dB].

    Returns:
        The figure and axes objects.
    """
    magnitude = np.abs(data)
    zero_to_smallest_float(magnitude)
    plot_data = 20 * np.log10(magnitude / magnitude.max())

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set_xlabel("Cross-Range [m]")
    ax.set_ylabel("Slant Range [km]")

    mesh = ax.pcolormesh(cross_range_axis, r_axis / 1e3, plot_data)
    mesh.set_clim(cbar_min, 0)
    cbar = fig.colorbar(mesh)
    cbar.set_label("Normalised Magnitude [dB]")

    fig.tight_layout()
    return fig, ax


def _plot_raw(r_axis: np.ndarray, data: np.ndarray, title: str) -> None:
    """Plots the magnitude of a range × slow-time matrix (debug helper)."""
    pulses = range(data.shape[1])
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    mesh = ax.pcolormesh(pulses, r_axis / 1e3, np.abs(data))
    ax.set_xlabel("Pulse Index (along-track)")
    ax.set_ylabel("Slant Range [km]")
    fig.colorbar(mesh, label="Magnitude")
    fig.tight_layout()
