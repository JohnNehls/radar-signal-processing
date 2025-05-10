import numpy as np
from scipy import fft
from . import constants as c
from .waveform_helpers import matchfilter_with_waveform


def range_axis(fs: float, Nr: int):
    """Create range labels for the fast-time axis"""
    dR_grid = c.C / (2 * fs)
    R_axis = np.arange(1, Nr + 1) * dR_grid  # Process fast time
    return R_axis


def number_range_bins(fs: float, prf: float):
    "Calculate the number of range bins in an RDM"
    # return round(fs / prf)
    return int(fs / prf)


def dataCube(fs: float, prf: float, Np: int):
    """Create an empty or noise datacube
    Outputs unprocessed datacube, both in fast and slow time
    inputs:
      fs = sampling frequency
      prf= pulse repitition frequncy of the radar
      Np = number of pulses in a CPI
    outputs:
      datacube of size (Nrange_bins, Np)
    """
    Nr = number_range_bins(fs, prf)
    dc = np.zeros((Nr, Np), dtype=np.complex64)
    return dc


def doppler_process(data_cube, fs):
    """Process data cube in place
    args:
        data_cube: time domain data to be Doppler processed
       fs: sample frequncy of the time domain data

    f_axis : [-PRF/2, PRF/2)
    r_axis : [delta_r, R_ambigious]
    """
    Nr, Np = data_cube.shape
    dR_grid = c.C / (2 * fs)
    PRF = fs / data_cube.shape[0]
    R_axis = np.arange(1, Nr + 1) * dR_grid  # Process fast time
    f_axis = fft.fftshift(fft.fftfreq(Np, 1/PRF))  # process slow time
    data_cube[:] = fft.fftshift(fft.fft(data_cube, axis=1), axes=1)
    return f_axis, R_axis


def matchfilter(dataCube, pulse_wvf, pedantic=True):
    """Inplace match filter on data cube"""
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
