#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
from rsp.waveform_helpers import matchfilter_with_waveform
from rsp.waveform import uncoded_pulse, barker_coded_pulse, BARKER_DICT
from rsp.waveform_helpers import zeropad_waveform

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True

print("#############################################")
print("Problem 2: Barker sidelobe check code example")
print("#############################################\n")

print("The sidelobes do not follow the table")
print("\tconsider the ypeak/nchips comment in the problem")

# constants
BW = 4e6  # Hz
sampleRate = 16e6  # Hz
Npad = 50

tu, mag_u = uncoded_pulse(sampleRate, BW)
tu, mag_u = zeropad_waveform(tu, mag_u, Npad)

fig, ax = plt.subplots(1, 2)
fig.suptitle("S3P2 Barker sidelobe check")
ax[0].plot(tu, mag_u, label="uncoded")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pulse [v]")
ax[0].set_title("normalized pulse")

iu, conv_u = matchfilter_with_waveform(mag_u, mag_u)
conv_u = abs(conv_u)
# conv_u = 20*np.log(conv_u)
ax[1].plot(iu, conv_u, "-x", label="uncoded")
ax[1].set_xlabel("index offset")
ax[1].set_xlabel("magnitude")
ax[1].set_title("xcorrelation")
# ax[1].set_yscale("log")

for nChip in BARKER_DICT.keys():
    t_b, mag_b = barker_coded_pulse(sampleRate, BW, nChip)
    ax[0].plot(t_b, mag_b, label=f"barker {nChip}")

    t_b, mag_b = zeropad_waveform(t_b, mag_b, Npad)
    ib, conv_b = matchfilter_with_waveform(mag_b, mag_b)
    conv_b = abs(conv_b)
    # conv_b = 20*np.log(conv_b)
    ax[1].plot(ib, conv_b, label=f"barker {nChip}")

for a in ax:
    a.grid()
    a.legend()

plt.show(block=BLOCK)
