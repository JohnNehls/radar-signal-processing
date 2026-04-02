import pytest
import rad_lab.pulse_doppler_radar as pdr

C = 3e8  # speed of light [m/s]


def test_range_unambiguous():
    assert pdr.range_unambiguous(50e3) == pytest.approx(C / (2 * 50e3))
    assert pdr.range_unambiguous(10e3) == pytest.approx(15000.0)


def test_range_aliased_no_wrap():
    # target within unambiguous range — should be unchanged
    assert pdr.range_aliased(1500.0, 50e3) == pytest.approx(1500.0)


def test_range_aliased_wraps():
    # unambiguous range at 50 kHz PRF = 3000 m; 15500 m aliases to 500 m
    assert pdr.range_aliased(15.5e3, 50e3) == pytest.approx(500.0)


def test_frequency_delta_doppler_approaching():
    # closing target (negative range rate) produces positive Doppler shift
    fd = pdr.frequency_delta_doppler(-750, 10e9)
    assert fd == pytest.approx(10e9 * 2 * 750 / C)


def test_frequency_delta_doppler_stationary():
    assert pdr.frequency_delta_doppler(0, 10e9) == pytest.approx(0.0)


def test_frequency_aliased_wraps_positive():
    # 50 kHz aliases to 2 kHz at 16 kHz PRF (50000 % 16000 = 2000, within +/- 8000)
    assert pdr.frequency_aliased(50e3, 16e3) == pytest.approx(2000.0)


def test_frequency_aliased_wraps_negative():
    # 12 kHz aliases to -4 kHz at 16 kHz PRF (12000 > 8000 → 12000 - 16000)
    assert pdr.frequency_aliased(12e3, 16e3) == pytest.approx(-4000.0)


def test_frequency_aliased_no_wrap():
    assert pdr.frequency_aliased(3e3, 16e3) == pytest.approx(3000.0)


def test_rangeRate_pm_unambiguous():
    # PRF=50 kHz, f0=10 GHz → 50e3 * 3e8 / (4 * 10e9) = 375 m/s
    assert pdr.range_rate_pm_unambiguous(50e3, 10e9) == pytest.approx(375.0)


def test_rangeRate_aliased_prf_f0_wraps():
    # PRF=16 kHz, f0=10 GHz → rangeRate_max=120 m/s; -750 m/s aliases to -30 m/s
    assert pdr.range_rate_aliased_prf_f0(-750, 16e3, 10e9) == pytest.approx(-30.0)


def test_rangeRate_aliased_prf_f0_no_wrap():
    # target well within unambiguous range rate
    assert pdr.range_rate_aliased_prf_f0(100.0, 50e3, 10e9) == pytest.approx(100.0)
