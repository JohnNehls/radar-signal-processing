import numpy as np
from scipy import fft
from . import constants as c
from .waveform_helpers import matchfilter_with_waveform


def range_axis(fs: float, N_r: int) -> np.ndarray:
    """Generates the range axis for a radar datacube.

    This function calculates the range corresponding to each range bin
    based on the sampling frequency. The range resolution is determined
    by the speed of light and the sampling rate.

    Args:
        fs (float): The sampling frequency in Hertz [Hz].
        N_r (int): The number of range bins (samples in fast-time).

    Returns:
        np.ndarray: A 1D NumPy array representing the range axis in meters [m].
    """
    dR_grid = c.C / (2 * fs)
    R_axis = np.arange(1, N_r + 1) * dR_grid  # Process fast time
    return R_axis


def number_range_bins(fs: float, prf: float) -> int:
    """Calculates the number of range bins.

    The number of range bins is determined by the number of samples collected
    during one pulse repetition interval (PRI). PRI is the reciprocal of the
    pulse repetition frequency (PRF).

    Args:
        fs (float): The sampling frequency [Hz].
        prf (float): The pulse repetition frequency [Hz].

    Returns:
        int: The total number of range bins.
    """
    return int(fs / prf)


def dataCube(fs: float, prf: float, N_p: int) -> np.ndarray:
    """Creates an empty, complex-valued datacube.

    This function initializes a 2D NumPy array (datacube) with zeros,
    representing the raw data collected over a coherent processing interval (CPI).
    The dimensions are determined by the number of range bins and the number of pulses.

    Args:
        fs (float): The sampling frequency in Hertz [Hz].
        prf (float): The pulse repetition frequency in Hertz [Hz].
        N_p (int): The number of pulses in the coherent processing interval (CPI).

    Returns:
        np.ndarray: A 2D NumPy array of shape (N_range_bins, N_pulses)
                    initialized with complex zeros.
    """
    Nr = number_range_bins(fs, prf)
    dc = np.zeros((Nr, N_p), dtype=np.complex64)
    return dc


def doppler_process(datacube: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Performs Doppler processing on a radar datacube.

    This function applies a Fast Fourier Transform (FFT) across the slow-time
    (pulse) dimension of the datacube to transform the data into the
    Range-Doppler domain. The operation is performed in-place on the input
    datacube. It also generates the corresponding Doppler frequency and range axes.

    Args:
        datacube (np.ndarray): A 2D NumPy array representing the time-domain
                             datacube, with shape (N_range_bins, N_pulses).
                             This array will be modified in-place.
        fs (float): The sampling frequency in Hertz [Hz].

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - f_axis (np.ndarray): The Doppler frequency axis, [-PRF/2, PRF/2) [Hz].
            - R_axis (np.ndarray): The range axis [delta_r, R_ambigious] [m].
    """
    N_r, N_p = datacube.shape
    dR_grid = c.C / (2 * fs)
    PRF = fs / datacube.shape[0]
    R_axis = np.arange(1, N_r + 1) * dR_grid  # Process fast time
    f_axis = fft.fftshift(fft.fftfreq(N_p, 1 / PRF))  # process slow time
    datacube[:] = fft.fftshift(fft.fft(datacube, axis=1), axes=1)
    return f_axis, R_axis


def matchfilter(datacube: np.ndarray, pulse_wvf: np.ndarray, pedantic: bool = True) -> None:
    """Applies a matched filter to a datacube for pulse compression.

    This function processes each pulse (column) in the datacube with a matched
    filter to perform pulse compression, which improves signal-to-noise ratio
    and range resolution. The operation is performed in-place.

    Two implementations are available:
    - Pedantic (True): Iteratively applies the matched filter to each pulse
      using a time-domain helper function. This is typically slower but can
      be clearer to understand.
    - Non-pedantic (False): Uses a more efficient frequency-domain approach
      by performing convolution via FFT. This involves a single FFT of the
      waveform kernel and is generally faster for large datacubes.

    Args:
        datacube (np.ndarray): The 2D time-domain datacube to be processed, with
                               shape (N_range_bins, N_pulses). This array is
                               modified in-place.
        pulse_wvf (np.ndarray): A 1D array representing the transmitted pulse
                                waveform samples.
        pedantic (bool, optional): If True, uses the iterative, time-domain
                                   filtering approach. If False, uses the
                                   faster frequency-domain convolution.
                                   Defaults to True.

    Returns:
        None: The `datacube` is modified in-place.
    """
    if pedantic:
        for j in range(datacube.shape[1]):
            _, mf = matchfilter_with_waveform(datacube[:, j], pulse_wvf)
            datacube[:, j] = mf
    else:
        # Take FFT convolution directly
        kernel = np.conj(pulse_wvf)[::-1]

        # Pad and "center" pulse relative to 0 index so output is centered (datacube is centered)
        # Method was tested but should be tested further
        # ref:  https://stackoverflow.com/questions/29746894/why-is-my-convolution-result-shifted-when-using-fft
        kernel = np.pad(kernel, pad_width=(0, datacube.shape[0] - pulse_wvf.size))
        offset = -int(pulse_wvf.size / 2)
        if offset % 2:
            offset += 1
        kernel = np.roll(kernel, offset)

        Kernel = fft.fft(kernel).reshape(datacube.shape[0], 1)
        PulseM = Kernel @ np.ones((1, datacube.shape[1]))
        DataCube = fft.fft(datacube, axis=0)
        datacube[:] = fft.ifft(PulseM * DataCube, axis=0, overwrite_x=True, workers=2)
