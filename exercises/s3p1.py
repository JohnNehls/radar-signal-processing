#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
sys.path.append("..")

from waveform_helpers import plotPulseAndSpectrum, plotPulseAndCrossCorrelation
from waveforms import makeLFMPulse, makeUncodedPulse, makeBarkerCodedPulse, makeRandomCodedPulse, makeLFMPulse

print("#############################################")
print("Problem 1: write the pulse functions and plot")
print("#############################################")

# constants
sampleRate = 10
BW = 1
output_time_T = 2 # time of time sample in terms of lenght of the pulse T

print("## uncoded example ##")
t_u, mag_u = makeUncodedPulse(sampleRate, BW, output_length_T=output_time_T, normalize=False)
plotPulseAndSpectrum(t_u, mag_u, f"S3P1 uncoded pulse {sampleRate=} {BW=}")
plotPulseAndCrossCorrelation(t_u, mag_u, f"uncoded pulse {sampleRate=} {BW=}")

print("## Barker example ##")
nChip = 7
t_b, mag_b = makeBarkerCodedPulse(sampleRate, BW, nChip, output_length_T=output_time_T,
                                      normalize=False)
plotPulseAndSpectrum(t_b, mag_b, f"S3P1 Barker coded pulse {nChip=} {sampleRate=} {BW=}")
plotPulseAndCrossCorrelation(t_b, mag_b, f"Barker coded pulse {nChip=} {sampleRate=} {BW=}")

print("## random code example  ##")
nChip = 7
t_r, mag_r = makeRandomCodedPulse(sampleRate, BW, nChip, output_length_T=output_time_T,
                                  normalize=False)
plotPulseAndSpectrum(t_r, mag_r, f"S3P1 random coded pulse {nChip=} {sampleRate=} {BW=}")
plotPulseAndCrossCorrelation(t_r, mag_r, f"S3P1 random coded pulse {nChip=} {sampleRate=} {BW=}")
print("## LFM example ##")
T = 2
chirpUpDown=1
t_lfm, mag_lfm = makeLFMPulse(sampleRate, BW, T, chirpUpDown, output_length_T=output_time_T,
                              normalize=False)
fig, ax = plotPulseAndSpectrum(t_lfm, mag_lfm,
                               f"S3P1 LFM pulse {chirpUpDown=} {T=}{sampleRate=} {BW=}")
plotPulseAndCrossCorrelation(t_lfm, mag_lfm,
                               f"S3P1 LFM pulse {chirpUpDown=} {T=}{sampleRate=} {BW=}")

plt.show()
