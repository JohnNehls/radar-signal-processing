import numpy as np
from scipy import fft
from . import constants as c
from .waveform_helpers import matchfilter_with_waveform


def range_axis(fs: float, N_r: int):
    """
    Create range labels for the fast-time axis.
    Args:
        fs (float) : Sample rate [Hz]
        N_r (int) : Number of range bins
    Return:
        R_axis : 1D array
    """
    dR_grid = c.C / (2 * fs)
    R_axis = np.arange(1, N_r + 1) * dR_grid  # Process fast time
    return R_axis


def number_range_bins(fs: float, prf: float):
    """
    Calculate the number of range bins in an RDM
    Args:
        fs (float) : Sample rate [Hz]
        prf (float) : Pulse Repition Interval [s]
    Return:
        N_r : (int)
    """
    return int(fs / prf)


def dataCube(fs: float, prf: float, N_p: int):
    """
    Create an empty datacube.
    Args:
        fs (float) : Sample rate [Hz]
        prf (float) : Pulse Repition Interval [s]
        N_p (int) : Number of pulses in the CPI
    Return:
        datacube : 2D array with shape (Nrange_bins, Np)
    """
    Nr = number_range_bins(fs, prf)
    dc = np.zeros((Nr, N_p), dtype=np.complex64)
    return dc


def doppler_process(datacube, fs: float):
    """Doppler process data cube in place.
    Args:
        datacube (2D array) : Time domain data to be Doppler processed
        fs (float) : Sample rate [Hz]
    Returns:
        None
    Notes:
        - New datacube axes:
            - f_axis : [-PRF/2, PRF/2)
            - r_axis : [delta_r, R_ambigious]
    """
    N_r, N_p = datacube.shape
    dR_grid = c.C / (2 * fs)
    PRF = fs / datacube.shape[0]
    R_axis = np.arange(1, N_r + 1) * dR_grid  # Process fast time
    f_axis = fft.fftshift(fft.fftfreq(N_p, 1 / PRF))  # process slow time
    datacube[:] = fft.fftshift(fft.fft(datacube, axis=1), axes=1)
    return f_axis, R_axis


def matchfilter(dataCube, pulse_wvf, pedantic=True):
    """
    Apply match filter to a datacube in place.
    Args:
        datacube (2D array) : Time domain data to be Doppler processed
        pulse_wvf (1d array) : Sample rate [Hz]
        pedantic (bool) : Pedantic algorithm flag
    Returns:
        None
    Notes:
        - Pedantic algorithm leaves the phase of the datacube zero for non-zero elements.
        - Non-pedantic algorithm is an attempt at the efficient match filter in frequency space, only needing to take the the fft of the waveform once.
    """
    if pedantic:
        for j in range(dataCube.shape[1]):
            _, mf = matchfilter_with_waveform(dataCube[:, j], pulse_wvf)
            dataCube[:, j] = mf
    else:
        # Take FFT convolution directly
        kernel = np.conj(pulse_wvf)[::-1]

        # Pad and "center" pulse relative to 0 index so output is centered (dataCube is centered)
        # Method was tested but should be tested further
        # ref:  https://stackoverflow.com/questions/29746894/why-is-my-convolution-result-shifted-when-using-fft
        kernel = np.pad(kernel, pad_width=(0, dataCube.shape[0] - pulse_wvf.size))
        offset = -int(pulse_wvf.size / 2)
        if offset % 2:
            offset += 1
        kernel = np.roll(kernel, offset)

        Kernel = fft.fft(kernel).reshape(dataCube.shape[0], 1)
        PulseM = Kernel @ np.ones((1, dataCube.shape[1]))
        DataCube = fft.fft(dataCube, axis=0)
        dataCube[:] = fft.ifft(PulseM * DataCube, axis=0, overwrite_x=True, workers=2)
