#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from rsp.waveform_helpers import (
    matchFilterPulse,
    unityVarianceComplexNoise,
    addWvfAtIndex,
    plotPulseAndSpectrum,
)
from rsp.waveform import makeUncodedPulse, makeBarkerCodedPulse, makeLFMPulse

plt.close("all")
print("##############################")
print("Problem 3: noisy xcorrelations")
print("##############################")

# constants for all waveforms
sampleRate = 20e6  # Hz
BW = 4e6  # Hz

print("## Case 1: add uncoded pulse ######")
print("\tcreate noise")
noise_1 = unityVarianceComplexNoise(1000)

print("verify uncoded pulse BW")
t_u, mag_u = makeUncodedPulse(sampleRate, BW, output_length_T=100, normalize=True)
plotPulseAndSpectrum(t_u, mag_u, "uncoded BW check", True)

# make pulse without extra points
t_u, mag_u = makeUncodedPulse(sampleRate, BW, output_length_T=1, normalize=True)
indx_1 = 200
SNR = 20  # noise is at 0dB
mag_u_s = 10 ** (SNR / 20) * mag_u

addWvfAtIndex(noise_1, mag_u_s, indx_1)  # add in place
print("verify noise is at ~ 0dB")
print(f"\t{10*np.log10(np.var(noise_1))=}")

mf, m_index_shift = matchFilterPulse(noise_1, mag_u)

fig, ax = plt.subplots(1, 3)
fig.suptitle(f"S3P3 case 1: uncoded pulse at index{indx_1}")
ax[0].plot(np.real(noise_1), "-o")
ax[0].set_title("noise")
ax[0].set_xlabel("sample")
ax[0].set_ylabel("real noise")
ax[1].plot(t_u, abs(mag_u_s), "-o")
ax[1].set_title("uncoded pulse")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("magnitude")
ax[2].plot(abs(mf), "-o")
ax[2].set_xlabel("sample")
ax[2].set_ylabel("matched magnitude")
ax[2].set_title("match filter on noise")
plt.tight_layout()
for a in ax:
    a.grid()

print("## Case 2: three uncoded pulse ####")
noise_2 = unityVarianceComplexNoise(1000)

# first pulse
indx_1 = 128
SNR = 15  # noise is at 0dB
_, mag_u = makeUncodedPulse(sampleRate, BW, output_length_T=1, normalize=True)
mag_u_s = 10 ** (SNR / 20) * mag_u
addWvfAtIndex(noise_2, mag_u_s, indx_1)  # add in place

# second pulse
indx_2 = 200
SNR = 30  # noise is at 0dB
_, mag_u = makeUncodedPulse(sampleRate, BW, output_length_T=1, normalize=True)
mag_u_s = 10 ** (SNR / 20) * mag_u
addWvfAtIndex(noise_2, mag_u_s, indx_2)  # add in place

# third pulse
indx_3 = 950
SNR = 20  # noise is at 0dB
_, mag_u = makeUncodedPulse(sampleRate, BW, output_length_T=1, normalize=True)
mag_u_s = 10 ** (SNR / 20) * mag_u
addWvfAtIndex(noise_2, mag_u_s, indx_3)  # add in place

mf, _ = matchFilterPulse(noise_2, mag_u)

fig, ax = plt.subplots(1, 4)
fig.suptitle("S3P3 case 2: three uncoded pulses, check SNR")
ax[0].plot(np.real(noise_2), "-o")
ax[0].set_title("noise")
ax[0].set_xlabel("sample")
ax[0].set_ylabel("real noise")
ax[1].plot(abs(mag_u_s), "-o")
ax[1].set_title("uncoded pulse")
ax[1].set_xlabel("sample")
ax[1].set_ylabel("magnitude")
ax[2].plot(abs(mf), "-o")
ax[2].set_xlabel("sample")
ax[2].set_ylabel("matched magnitude")
ax[2].set_title("match filter on noise")
ax[3].plot(10 * np.log(abs(mf)), "-o")
ax[3].set_xlabel("sample")
ax[3].set_ylabel("matched dB")
ax[3].set_title("match filter on noise")
plt.tight_layout()
for a in ax:
    a.grid()


print("## Case 3: LFM and BPSK pulses ####")
noise_3 = unityVarianceComplexNoise(1000)

print(f"{10*np.log10(np.var(noise_3))=}")  # verify noise is at 0dB
# LFM
lfm_idx = 300
SNR = 20
T = 2e-6
chirpUpDown = 1
_, mag_lfm = makeLFMPulse(sampleRate, BW, T, chirpUpDown, output_length_T=1, normalize=True)
mag_lfm_s = 10 ** (SNR / 20) * mag_lfm
addWvfAtIndex(noise_3, mag_lfm_s, lfm_idx)  # add in place

# BPSK
bpsk_idx = 600
SNR = 20
_, mag_b = makeBarkerCodedPulse(sampleRate, BW, 13, output_length_T=1, normalize=True)
mag_b_s = 10 ** (SNR / 20) * mag_b
addWvfAtIndex(noise_3, mag_b_s, bpsk_idx)  # add in place

fig, ax = plt.subplots(1, 5)
fig.suptitle("S3P3 case 3")
# ax[0].plot(noise_3)
ax[0].plot(abs(noise_3))
ax[0].set_title("signal + noise")
ax[0].set_xlabel("sample")
# ax[1].plot(mag_lfm_s,'-o')
ax[1].plot(abs(mag_lfm_s), "-o")
ax[1].set_title(f"lfm pulse, index={lfm_idx}")
ax[1].set_xlabel("sample")
# ax[2].plot(mag_b_s,'-o')
ax[2].plot(abs(mag_b_s), "-o")
ax[2].set_title(f"bpsk pulse, index={bpsk_idx}")
ax[2].set_xlabel("sample")
ax[3].plot(abs(matchFilterPulse(noise_3, mag_lfm)[0]))
ax[3].set_title("lfm match")
ax[3].set_xlabel("sample")
ax[4].plot(abs(matchFilterPulse(noise_3, mag_b)[0]))
ax[4].set_title("bpsk match")
ax[4].set_xlabel("sample")
plt.tight_layout()
for a in ax:
    a.grid()


print("## Case 4: add uncoded pulse ####")
BW = 4e6  # Hz
SNR = 20

sampleRate = 16e6  # Hz
tb, mag_b = makeBarkerCodedPulse(sampleRate, BW, 13, output_length_T=1, normalize=True)
mag_b_s = 10 ** (SNR / 20) * mag_b
tu, mag_u = makeUncodedPulse(sampleRate, BW, output_length_T=13, normalize=True)
mag_u_s = 10 ** (SNR / 20) * mag_u

fig, ax = plt.subplots(1, 2)
fig.suptitle("Sec 2 prob 4 : compare uncoded to Barker 13 pulse")
ax[0].plot(tu, mag_u_s, "-o", label="uncoded")
ax[0].plot(tb, mag_b_s, "-x", label="barker13")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pulse amplitude [v]")
ax[0].legend()

conv_u, iu = matchFilterPulse(mag_u_s, mag_u)
conv_b, ib = matchFilterPulse(mag_b_s, mag_b)
ax[1].plot(iu, conv_u, "-o", label="uncoded")
ax[1].plot(ib, conv_b, "-x", label="barker13")
ax[1].set_xlabel("index shift")
ax[1].set_ylabel("matched filter")
ax[1].legend()
plt.tight_layout()
for a in ax:
    a.grid()

plt.show()
