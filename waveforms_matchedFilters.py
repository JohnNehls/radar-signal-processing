#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
from scipy import fft

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


#def main():
from waveform_helpers import plotPulseAndSpectrum, plotPulseAndCrossCorrelation

sampleRate = 100
BW = 10
output_time_T = 2 # time of time sample in terms of lenght of the pulse T,
print("#############################################")
print("Problem 1: write the pulse functions and plot")
print("#############################################")
####
print("## uncoded example ##")
t_u, mag_u = makeUncodedPulse(sampleRate, BW, output_length_T=output_time_T, normalize=False)
plotPulseAndSpectrum(t_u, mag_u, f"uncoded pulse {sampleRate=} {BW=}")
plotPulseAndCrossCorrelation(t_u, mag_u, f"uncoded pulse {sampleRate=} {BW=}")

print("## Barker example ##")
nChip = 7
t_b, mag_b = makeBarkerCodedPulse(sampleRate, BW, nChip, output_length_T=output_time_T,
                                      normalize=False)
plotPulseAndSpectrum(t_b, mag_b, f"Barker coded pulse {nChip=} {sampleRate=} {BW=}")
plotPulseAndCrossCorrelation(t_b, mag_b, f"Barker coded pulse {nChip=} {sampleRate=} {BW=}")

print("## random code example  ##")
nChip = 7
t_r, mag_r = makeRandomCodedPulse(sampleRate, BW, nChip, output_length_T=output_time_T,
                                  normalize=False)
plotPulseAndSpectrum(t_r, mag_r, f"random coded pulse {nChip=} {sampleRate=} {BW=}")

print("## LFM example ##")
T = 5
chirpUpDown=1
t_lfm, mag_lfm = makeLFMPulse(sampleRate, BW, T, chirpUpDown, output_length_T=output_time_T,
                              normalize=False)
fig, ax = plotPulseAndSpectrum(t_lfm, mag_lfm, f"LFM pulse {chirpUpDown=} {T=}{sampleRate=} {BW=}")

print("#############################################")
print("Problem 2: Barker sidelobe check code example")
print("#############################################")
print("The sidelobes do not follow the table")
print("\tconsider the ypeak/nchips comment in the problem")
output_time_T = 6 # increase time to increase DFT resolution
for nChip in barker_dict.keys():
    t_b, mag_b = makeBarkerCodedPulse(sampleRate, BW, nChip, output_length_T=output_time_T,
                                      normalize=False)
    # plotPulseAndSpectrum(t_b, mag_b, f"Barker coded pulse {nChip=} {sampleRate=} {BW=}")
    plotPulseAndCrossCorrelation(t_b, mag_b, f"Barker coded pulse {nChip=} {sampleRate=} {BW=}")


print("##############################")
print("Problem 3: noisy xcorrelations")
print("##############################")
