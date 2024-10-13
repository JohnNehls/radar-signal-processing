#!/usr/bin/env python

from rsp.waveform import process_waveform_dict

bw = 10e6

wvf = {"type": "lfm", "bw": bw, "T": 10 / 40e6, "chirpUpDown": 1}

radar = {
    "fcar": 10e9,
    "txPower": 1e3,
    "txGain": 10 ** (30 / 10),
    "rxGain": 10 ** (30 / 10),
    "opTemp": 290,
    "sampRate": 2 * bw,
    "noiseFactor": 10 ** (8 / 10),
    "totalLosses": 10 ** (8 / 10),
    "PRF": 200e3,
    "dwell_time": 2e-3,
}

process_waveform_dict(wvf, radar)

print(wvf)
