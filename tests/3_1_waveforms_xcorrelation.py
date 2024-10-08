#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
from rsp.waveform_helpers import plot_pulse_and_spectrum, plot_pulse_and_xcorrelation
from rsp.waveform import uncoded_pulse, barker_coded_pulse, random_coded_pulse, lfm_pulse

# Can turn blocking plot off in the commandline
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

print("#############################################")
print("Problem 1: write the pulse functions and plot")
print("#############################################")

# constants
sampleRate = 10
BW = 1
output_time_T = 2  # time of time sample in terms of lenght of the pulse T

print("## uncoded example ##")
t_u, mag_u = uncoded_pulse(sampleRate, BW, output_length_T=output_time_T, normalize=False)
plot_pulse_and_spectrum(t_u, mag_u, f"S3P1 uncoded pulse {sampleRate=} {BW=}")
plot_pulse_and_xcorrelation(t_u, mag_u, f"uncoded pulse {sampleRate=} {BW=}")

print("## Barker example ##")
nChip = 7
t_b, mag_b = barker_coded_pulse(
    sampleRate, BW, nChip, output_length_T=output_time_T, normalize=False
)
plot_pulse_and_spectrum(t_b, mag_b, f"S3P1 Barker coded pulse {nChip=} {sampleRate=} {BW=}")
plot_pulse_and_xcorrelation(t_b, mag_b, f"Barker coded pulse {nChip=} {sampleRate=} {BW=}")

print("## random code example  ##")
nChip = 7
t_r, mag_r = random_coded_pulse(
    sampleRate, BW, nChip, output_length_T=output_time_T, normalize=False
)
plot_pulse_and_spectrum(t_r, mag_r, f"S3P1 random coded pulse {nChip=} {sampleRate=} {BW=}")
plot_pulse_and_xcorrelation(t_r, mag_r, f"S3P1 random coded pulse {nChip=} {sampleRate=} {BW=}")
print("## LFM example ##")
T = 2
chirpUpDown = 1
t_lfm, mag_lfm = lfm_pulse(
    sampleRate, BW, T, chirpUpDown, output_length_T=output_time_T, normalize=False
)
fig, ax = plot_pulse_and_spectrum(
    t_lfm, mag_lfm, f"S3P1 LFM pulse {chirpUpDown=} {T=}{sampleRate=} {BW=}"
)
plot_pulse_and_xcorrelation(
    t_lfm, mag_lfm, f"S3P1 LFM pulse {chirpUpDown=} {T=}{sampleRate=} {BW=}"
)
plt.show(block=BLOCK)
