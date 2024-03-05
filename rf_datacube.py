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

def create_dataCube(fs, PRF, Np):
    """data cube
    ouputs unprocessed datacube, both in fast and slow time
    """
    Nr = round(fs/PRF)
    dc = np.zeros((Nr,Np),dtype=np.complex64)

    return dc

def dopplerProcess_dataCube(dc, fs, PRF):
    """data cube
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

    for i in range(dc.shape[0]):
        dc[i,:] = fft.fftshift(fft.fft(dc[i]))

    return dc, f_axis, R_axis


def R_pf_tgt(pf_pos : list, pf_vel : list, tgt_pos : list, tgt_vel : list):

    R_vec = np.array([tgt_pos[0] - pf_pos[0], tgt_pos[1] - pf_pos[1], tgt_pos[2] - pf_pos[2]])
    R_unit_vec = R_vec/norm(R_vec)

    R_mag = np.sqrt(R_vec[0]**2 + R_vec[1]**2 + R_vec[2]**2)

    R_dot = np.dot(tgt_vel, R_unit_vec) - np.dot(pf_vel, R_unit_vec)

    return R_vec, R_mag, R_dot
