import numpy as np
import pytest
import rsp.uniform_linear_arrays as ula
import rsp.monopulse as mp
from rsp.noise import unity_variance_complex_noise

N_SAMPLES = 1000
ARRAY_POS = np.array([-1 / 4, 1 / 4])  # element positions in wavelengths
DX = ARRAY_POS[1] - ARRAY_POS[0]


def _make_received(tgt_angle, snr_db, seed):
    """Return (sig_a, sig_b) for a single-frequency signal at tgt_angle."""
    np.random.seed(seed)
    time_ar = np.linspace(0, 10, N_SAMPLES)
    signal_ar = np.exp(1j * 2 * np.pi * 1 * time_ar)
    steer_vec = ula.steering_vector(ARRAY_POS, tgt_angle)
    snr_volt = 10 ** (snr_db / 20)
    received = [
        snr_volt * sv * signal_ar + unity_variance_complex_noise(N_SAMPLES) for sv in steer_vec
    ]
    return received[0], received[1]


def _noiseless_received(tgt_angle):
    """Return (sig_a, sig_b) with no noise."""
    time_ar = np.linspace(0, 10, N_SAMPLES)
    signal_ar = np.exp(1j * 2 * np.pi * 1 * time_ar)
    steer_vec = ula.steering_vector(ARRAY_POS, tgt_angle)
    return steer_vec[0] * signal_ar, steer_vec[1] * signal_ar


# ---------------------------------------------------------------------------
# amplitude_monopulse
# ---------------------------------------------------------------------------


def test_amplitude_monopulse_boresight_is_zero():
    sig_a, sig_b = _noiseless_received(0.0)
    v_theta = mp.amplitude_monopulse(sig_a, sig_b, DX)
    assert np.allclose(v_theta, 0.0, atol=1e-10)


@pytest.mark.parametrize("angle", [-10, -5, 0, 5, 10])
def test_amplitude_monopulse_noiseless_constant(angle):
    # Noiseless single-frequency signal: all samples must give the same estimate
    sig_a, sig_b = _noiseless_received(angle)
    v_theta = mp.amplitude_monopulse(sig_a, sig_b, DX)
    assert np.allclose(v_theta, v_theta[0], atol=1e-10)


def test_amplitude_monopulse_sign():
    sig_a_pos, sig_b_pos = _noiseless_received(5.0)
    sig_a_neg, sig_b_neg = _noiseless_received(-5.0)
    v_pos = mp.amplitude_monopulse(sig_a_pos, sig_b_pos, DX)
    v_neg = mp.amplitude_monopulse(sig_a_neg, sig_b_neg, DX)
    assert np.all(v_pos > 0)
    assert np.all(v_neg < 0)


# ---------------------------------------------------------------------------
# monopulse_angle_deg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [-5, 0, 5])
def test_monopulse_angle_deg_noiseless_within_half_degree(angle):
    # Monopulse is an approximation; within ±5° the error should be < 0.5°
    sig_a, sig_b = _noiseless_received(angle)
    measured = mp.monopulse_angle_deg(sig_a, sig_b, DX)
    assert np.allclose(measured, angle, atol=0.5)


def test_angle_error_small_at_high_snr():
    # At SNR=30 dB the monopulse estimate should be accurate to within 0.5 degrees
    sig_a, sig_b = _make_received(tgt_angle=2, snr_db=30, seed=0)
    error = np.mean(abs(mp.monopulse_angle_deg(sig_a, sig_b, DX) - 2))
    assert error < 0.5


def test_angle_error_decreases_with_snr():
    # Mean angle error at high SNR should be less than at low SNR
    tgt_angle = 2
    high_snr_a, high_snr_b = _make_received(tgt_angle, snr_db=30, seed=10)
    low_snr_a, low_snr_b = _make_received(tgt_angle, snr_db=-5, seed=10)
    error_high = np.mean(abs(mp.monopulse_angle_deg(high_snr_a, high_snr_b, DX) - tgt_angle))
    error_low = np.mean(abs(mp.monopulse_angle_deg(low_snr_a, low_snr_b, DX) - tgt_angle))
    assert error_high < error_low


# ---------------------------------------------------------------------------
# monopulse_angle_at_peak_deg
# ---------------------------------------------------------------------------


def test_monopulse_angle_at_peak_noiseless():
    # 1-D noiseless signal: peak-based estimate should be within 0.5° of true angle
    tgt_angle = 5.0
    sig_a, sig_b = _noiseless_received(tgt_angle)
    at_peak = mp.monopulse_angle_at_peak_deg(sig_a, sig_b, DX)
    assert abs(at_peak - tgt_angle) < 0.5


def test_monopulse_angle_at_peak_consistent_with_angle_deg():
    # monopulse_angle_at_peak_deg must return the same value as indexing
    # monopulse_angle_deg at the argmax of sig_a
    tgt_angle = 7.0
    sig_a, sig_b = _noiseless_received(tgt_angle)
    # Reshape to 2-D to exercise unravel_index path
    sig_a_2d = sig_a.reshape(20, 50)
    sig_b_2d = sig_b.reshape(20, 50)
    at_peak = mp.monopulse_angle_at_peak_deg(sig_a_2d, sig_b_2d, DX)
    theta_2d = mp.monopulse_angle_deg(sig_a_2d, sig_b_2d, DX)
    peak_idx = np.unravel_index(np.argmax(np.abs(sig_a_2d)), sig_a_2d.shape)
    assert at_peak == theta_2d[peak_idx]


def test_monopulse_angle_at_peak_returns_scalar():
    sig_a, sig_b = _noiseless_received(3.0)
    result = mp.monopulse_angle_at_peak_deg(sig_a, sig_b, DX)
    assert isinstance(result, float)
