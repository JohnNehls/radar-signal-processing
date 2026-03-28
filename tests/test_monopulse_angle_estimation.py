import numpy as np
import pytest
import rsp.uniform_linear_arrays as ula
from rsp.noise import unity_variance_complex_noise

TGT_ANGLE = 2  # degrees
N_SAMPLES = 1000
ARRAY_POS = np.array([-1 / 4, 1 / 4])  # element positions in wavelengths

def _monopulse_mean_error_deg(snr_db, seed):
    np.random.seed(seed)
    time_ar = np.linspace(0, 10, N_SAMPLES)
    signal_ar = np.exp(1j * 2 * np.pi * 1 * time_ar)

    steer_vec = ula.steering_vector(ARRAY_POS, TGT_ANGLE)
    snr_volt = 10 ** (snr_db / 20)

    received = [
        snr_volt * sv * signal_ar + unity_variance_complex_noise(N_SAMPLES)
        for sv in steer_vec
    ]

    dx = ARRAY_POS[1] - ARRAY_POS[0]
    rho = 2 * np.pi * dx
    sum_ch = received[0] + received[1]
    delta_ch = received[0] - received[1]
    v_theta = np.arctan(2 * (delta_ch / sum_ch).imag) / rho
    measured_theta = np.arcsin(v_theta)

    return np.mean(abs(np.rad2deg(measured_theta) - TGT_ANGLE))

def test_angle_error_decreases_with_snr():
    # Mean angle error should increase monotonically as SNR decreases
    test_snr_list = [_monopulse_mean_error_deg(snr_db=snr, seed=10)
                     for snr in np.arange(30, -10, -1)]
    assert test_snr_list == sorted(test_snr_list), f"{test_snr_list}"

def test_angle_error_small_at_high_snr():
    # At SNR=30 dB the monopulse estimate should be accurate to within 0.5 degrees
    error = _monopulse_mean_error_deg(snr_db=30, seed=0)
    assert error < 0.5
