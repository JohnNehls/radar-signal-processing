#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from scipy import fft
from scipy import signal
import matplotlib.pyplot as plt

from waveform_helpers import matchFilterPulse

# constants
C = 3e8
PI = np.pi

def calc_range_axis(fs, Nr):
    dR_grid = C/(2*fs)
    R_axis = np.arange(1,Nr+1)*dR_grid # Process fast time
    return R_axis

def calc_number_range_bins(fs, prf):
    return round(fs/prf)

def create_dataCube(fs, prf, Np, noise=False):
    """data cube
    Outputs unprocessed datacube, both in fast and slow time
    inputs:
      Nr = number of range bins
      Np = number of pulses
    outputs:
      Datacube of size [Nr,Np]
    """
    Nr = calc_number_range_bins(fs, prf)

    if noise:
        dc = (np.random.randn(Nr,Np) + 1j*np.random.randn(Nr,Np))/np.sqrt(2*Np)
    else:
        dc = np.zeros((Nr,Np),dtype=np.complex64)

    return dc

def dopplerProcess_dataCube(dc, fs, PRF):
    """Process data cube in place
    ouputs:\n
    dataCube : \n
    f_axis : [-fs/2, fs/2)\n
    r_axis : [delta_r, R_ambigious]\n
    """
    Np = dc.shape[1]

    dR_grid = C/(2*fs)
    Nr = round(fs/PRF)
    R_axis = np.arange(1,Nr+1)*dR_grid # Process fast time
    f_axis = fft.fftshift(fft.fftfreq(Np,1/fs)) # process slow time

    dc[:] = fft.fftshift(fft.fft(dc, axis=1), axes=1)

    return f_axis, R_axis

def applyMatchFilterToDataCube(dataCube, pulse_wvf, pedantic=True):
    """Inplace match filter on data cube"""
    if pedantic:
        for j in range(dataCube.shape[1]):
            mf, _= matchFilterPulse(dataCube[:,j], pulse_wvf)
            dataCube[:, j] = mf
    else:
        # Take FFT convolution directly
        kernel = np.conj(pulse_wvf)[::-1]

        # Pad and "center" pulse relative to 0 index so output is centered (dataCube is centered)
        # Method was tested but should be tested further
        # ref:  https://stackoverflow.com/questions/29746894/why-is-my-convolution-result-shifted-when-using-fft
        kernel = np.pad(kernel, pad_width=(0,dataCube.shape[0]-pulse_wvf.size))
        offset = -int(pulse_wvf.size/2)
        if offset%2:
            offset += 1
        kernel = np.roll(kernel, offset)

        Kernel = fft.fft(kernel).reshape(dataCube.shape[0],1)
        PulseM = Kernel@np.ones((1, dataCube.shape[1]))
        DataCube = fft.fft(dataCube, axis=0)
        dataCube[:] = fft.ifft(PulseM * DataCube, axis=0, overwrite_x=True, workers=2)


def R_pf_tgt(pf_pos : list, pf_vel : list, tgt_pos : list, tgt_vel : list):

    R_vec = np.array([tgt_pos[0] - pf_pos[0], tgt_pos[1] - pf_pos[1], tgt_pos[2] - pf_pos[2]])
    R_unit_vec = R_vec/norm(R_vec)

    R_mag = np.sqrt(R_vec[0]**2 + R_vec[1]**2 + R_vec[2]**2)

    R_dot = np.dot(tgt_vel, R_unit_vec) - np.dot(pf_vel, R_unit_vec)

    return R_vec, R_mag, R_dot
