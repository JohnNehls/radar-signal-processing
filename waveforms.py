#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm

# constants
C = 3e8
PI =  3.141592653589793
barker_dict = {2 : [ 1,-1],  # could also be [ 1, 1]
               3 : [ 1, 1,-1],
               4 : [ 1, 1,-1, 1],
               5 : [ 1, 1, 1,-1, 1],
               7 : [ 1, 1, 1,-1,-1, 1,-1],
               11: [ 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1],
               13: [ 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1]}

# Notes
# - need to fill in all function doc strings with input explinations

def makeUncodedPulse(sampleRate, BW, output_length_T=1, t_start=0, normalize=True):
    """baseband uncoded pulse"""
    assert output_length_T >= 1, "Error: must output a full pulse"
    assert sampleRate/BW >= 2, "Error: sample rate below Nyquist"

    T = 1/BW
    dt = 1/sampleRate
    t = np.arange(t_start, t_start + output_length_T*T + dt, dt)
    mag = np.zeros(t.size)

    mag[ np.where((t >= t_start) & (t <= T + t_start) ) ] = 1

    if normalize:
        mag = mag/norm(mag)

    return t, mag

def makeCodedPulse(sampleRate, BW, code, output_length_T=1, t_start=0, normalize=True):
    """baseband coded pulse"""
    assert output_length_T >= 1, "Error: must output a full pulse"

    nChips = len(code)

    Tc = 1/BW
    T = nChips*Tc
    dt = 1/sampleRate

    samplesPerChip = round(Tc*sampleRate)

    mag = np.zeros((nChips, samplesPerChip))

    for i, val in enumerate(code):
        mag[i,:] = val

    mag = mag.flatten()

    if output_length_T > 1:
        tmp = np.zeros(round(output_length_T*T*sampleRate))
        tmp[:mag.size] = mag
        mag = tmp

    t = np.arange(mag.size)*dt + t_start

    if normalize:
        mag = mag/norm(mag)

    return t, mag


def makeBarkerCodedPulse(sampleRate, BW, nChips, output_length_T=1, t_start=0, normalize=True):
    """baseband Barker coded pulse"""
    assert nChips in barker_dict, f"Error: {nChips=} is not a valid Barker code."
    assert nChips == len(barker_dict[nChips]), f"Error: Barker dict is incorrect"
    return makeCodedPulse(sampleRate, BW, barker_dict[nChips], output_length_T=output_length_T,
                          t_start=t_start, normalize=normalize)


def makeRandomCodedPulse(sampleRate, BW, nChips, output_length_T=1, t_start=0, normalize=True):
    """baseband Barker coded pulse"""
    code_rand = np.random.choice([1,-1], size=nChips)
    return makeCodedPulse(sampleRate, BW, code_rand, output_length_T=output_length_T,
                          t_start=t_start, normalize=normalize)


def makeLFMPulse(sampleRate, BW, T, chirpUpDown, output_length_T=1, t_start=0, normalize=True):
    """baseband LFM pulse"""
    assert output_length_T >= 1, "Error: must output a full pulse"
    dt = 1/sampleRate
    t = np.arange(t_start, t_start + output_length_T*T + dt, dt)
    f = chirpUpDown*(-BW*t+BW*t**2/T)/2
    mag = np.zeros(t.size,dtype=np.complex64)
    i_ar = np.where((t >= t_start) & (t <= (t_start + T)))
    mag[i_ar] = np.exp(1j*2*PI*f[i_ar])

    if normalize:
        mag = mag/norm(mag)

    return t, mag
