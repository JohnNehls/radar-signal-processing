import numpy as np
import pytest
from rsp.waveform_helpers import matchfilter_with_waveform, add_waveform_at_index
from rsp.noise import unity_variance_complex_noise
from rsp.waveform import uncoded_pulse, barker_coded_pulse, lfm_pulse

SAMPLE_RATE = 20e6
BW = 4e6


def _inject_and_filter(noise, waveform, inject_idx, snr_db):
    scaled = 10 ** (snr_db / 20) * waveform
    add_waveform_at_index(noise, scaled, inject_idx)
    _, mf = matchfilter_with_waveform(noise, waveform)
    return mf


def test_uncoded_pulse_peak_at_injected_index():
    np.random.seed(0)
    noise = unity_variance_complex_noise(1000)
    _, waveform = uncoded_pulse(SAMPLE_RATE, BW)
    inject_idx = 200
    mf = _inject_and_filter(noise, waveform, inject_idx, snr_db=20)
    assert np.argmax(abs(mf)) == inject_idx + len(waveform) // 2


def test_highest_snr_pulse_dominates():
    # Three uncoded pulses at different SNRs; the 30 dB one should set the peak
    np.random.seed(1)
    noise = unity_variance_complex_noise(1000)
    _, waveform = uncoded_pulse(SAMPLE_RATE, BW)
    for idx, snr in [(128, 15), (200, 30), (950, 20)]:
        add_waveform_at_index(noise, 10 ** (snr / 20) * waveform, idx)
    _, mf = matchfilter_with_waveform(noise, waveform)
    assert np.argmax(abs(mf)) == 200 + len(waveform) // 2


def test_lfm_matched_filter_peak():
    np.random.seed(2)
    noise = unity_variance_complex_noise(1000)
    _, waveform = lfm_pulse(SAMPLE_RATE, BW, T=2e-6, chirpUpDown=1)
    inject_idx = 300
    mf = _inject_and_filter(noise, waveform, inject_idx, snr_db=20)
    assert np.argmax(abs(mf)) == inject_idx + len(waveform) // 2


def test_barker13_matched_filter_peak():
    np.random.seed(3)
    noise = unity_variance_complex_noise(1000)
    _, waveform = barker_coded_pulse(SAMPLE_RATE, BW, 13)
    inject_idx = 600
    mf = _inject_and_filter(noise, waveform, inject_idx, snr_db=20)
    assert np.argmax(abs(mf)) == inject_idx + len(waveform) // 2
