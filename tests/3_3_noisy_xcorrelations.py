#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from rsp.waveform_helpers import (
    matchfilter_with_waveform,
    add_waveform_at_index,
    plot_pulse_and_spectrum,
)
from rsp.noise import unity_variance_complex_noise
from rsp.waveform import uncoded_pulse, barker_coded_pulse, lfm_pulse

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

print("##############################")
print("Problem 3: noisy xcorrelations")
print("##############################")

# NOTES
# - Matchfilter peaks show where the center of the pulse is (not the leading edge)

# constants for all waveforms
sampleRate = 20e6  # Hz
BW = 4e6  # Hz

print("## Case 1: add uncoded pulse ######")
print("\tcreate noise")
noise_1 = unity_variance_complex_noise(1000)

print("verify uncoded pulse BW")
t_u, mag_u = uncoded_pulse(sampleRate, BW, normalize=True)
plot_pulse_and_spectrum(t_u, mag_u, "uncoded BW check", Npad=1000)

# make pulse without extra points
t_u, mag_u = uncoded_pulse(sampleRate, BW)
indx_1 = 200
SNR = 20  # noise is at 0dB
mag_u_s = 10 ** (SNR / 20) * mag_u

add_waveform_at_index(noise_1, mag_u_s, indx_1)  # add in place
print("verify noise is at ~ 0dB")
print(f"\t{10*np.log10(np.var(noise_1))=}")

m_index_shift, mf = matchfilter_with_waveform(noise_1, mag_u)

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
noise_2 = unity_variance_complex_noise(1000)

# first pulse
indx_1 = 128
SNR = 15  # noise is at 0dB
_, mag_u = uncoded_pulse(sampleRate, BW)
mag_u_s = 10 ** (SNR / 20) * mag_u
add_waveform_at_index(noise_2, mag_u_s, indx_1)  # add in place

# second pulse
indx_2 = 200
SNR = 30  # noise is at 0dB
_, mag_u = uncoded_pulse(sampleRate, BW)
mag_u_s = 10 ** (SNR / 20) * mag_u
add_waveform_at_index(noise_2, mag_u_s, indx_2)  # add in place

# third pulse
indx_3 = 950
SNR = 20  # noise is at 0dB
_, mag_u = uncoded_pulse(sampleRate, BW)
mag_u_s = 10 ** (SNR / 20) * mag_u
add_waveform_at_index(noise_2, mag_u_s, indx_3)  # add in place

_, mf = matchfilter_with_waveform(noise_2, mag_u)

fig, ax = plt.subplots(1, 4)
fig.suptitle(f"S3P3 case 2: uncoded pulses @ index: {indx_1}, {indx_2}, {indx_3} -- check SNR")
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
noise_3 = unity_variance_complex_noise(1000)

print(f"{10*np.log10(np.var(noise_3))=}")  # verify noise is at 0dB
# LFM
lfm_idx = 300
SNR = 20
T = 2e-6
chirpUpDown = 1
_, mag_lfm = lfm_pulse(sampleRate, BW, T, chirpUpDown)
mag_lfm_s = 10 ** (SNR / 20) * mag_lfm
add_waveform_at_index(noise_3, mag_lfm_s, lfm_idx)  # add in place

# BPSK
bpsk_idx = 600
SNR = 20
_, mag_b = barker_coded_pulse(sampleRate, BW, 13)
mag_b_s = 10 ** (SNR / 20) * mag_b
add_waveform_at_index(noise_3, mag_b_s, bpsk_idx)  # add in place

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
ax[3].plot(abs(matchfilter_with_waveform(noise_3, mag_lfm)[1]))
ax[3].set_title("lfm match")
ax[3].set_xlabel("sample")
ax[4].plot(abs(matchfilter_with_waveform(noise_3, mag_b)[1]))
ax[4].set_title("bpsk match")
ax[4].set_xlabel("sample")
plt.tight_layout()
for a in ax:
    a.grid()

plt.show(block=BLOCK)
