#!/usr/bin/env python

import matplotlib.pyplot as plt
from rsp.waveform_helpers import matchfilter_with_waveform
from rsp.waveform import uncoded_pulse, barker_coded_pulse

plt.close("all")
print("##############################")
print("Problem 4: Compare Barker 13 to uncoded pulse")
print("##############################")

print("## Case 4: add uncoded pulse ####")
BW = 4e6  # Hz
SNR = 20

sampleRate = 16e6  # Hz
tb, mag_b = barker_coded_pulse(sampleRate, BW, 13, output_length_T=1, normalize=True)
mag_b_s = 10 ** (SNR / 20) * mag_b
tu, mag_u = uncoded_pulse(sampleRate, BW, output_length_T=13, normalize=True)
mag_u_s = 10 ** (SNR / 20) * mag_u

fig, ax = plt.subplots(1, 2)
fig.suptitle("Sec 2 prob 4 : compare uncoded to Barker 13 pulse")
ax[0].plot(tu, mag_u_s, "-o", label="uncoded")
ax[0].plot(tb, mag_b_s, "-x", label="barker13")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pulse amplitude [v]")
ax[0].legend()

conv_u, iu = matchfilter_with_waveform(mag_u_s, mag_u)
conv_b, ib = matchfilter_with_waveform(mag_b_s, mag_b)
ax[1].plot(iu, conv_u, "-o", label="uncoded")
ax[1].plot(ib, conv_b, "-x", label="barker13")
ax[1].set_xlabel("index shift")
ax[1].set_ylabel("matched filter")
ax[1].legend()
plt.tight_layout()
for a in ax:
    a.grid()

plt.show()
