import numpy as np
import pytest
import rsp.uniform_linear_arrays as ula
import rsp.monopulse as mp
from rsp import rdm
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import lfm_waveform
from rsp.returns import Target, Return

BW = 10e6

RADAR = Radar(
    fcar=10e9,
    txPower=1e3,
    txGain=10 ** (30 / 10),
    rxGain=10 ** (30 / 10),
    opTemp=290,
    sampRate=2 * BW,
    noiseFactor=10 ** (8 / 10),
    totalLosses=10 ** (8 / 10),
    PRF=50e3,
    dwell_time=2e-3,
)

WAVEFORM = lfm_waveform(BW, T=10 / 40e6, chirpUpDown=1)
TGT_ANGLE = 5  # degrees
DX = 1 / 2  # element separation in wavelengths


def test_monopulse_rdm_angle_error_within_threshold():
    array_pos = np.array([-DX / 2, DX / 2])
    steer_vec = ula.steering_vector(array_pos, TGT_ANGLE)

    dc_list = []
    for i, sv in enumerate(steer_vec):
        return_list = [Return(target=Target(range=2.4e3, rangeRate=0.2e3, rcs=10, sv=sv))]
        _, _, total_dc, _ = rdm.gen(RADAR, WAVEFORM, return_list, snr=True, debug=False, plot=False, seed=i)
        dc_list.append(total_dc)

    measured_theta = mp.monopulse_angle_at_peak_deg(dc_list[0], dc_list[1], DX)
    error = abs(measured_theta - TGT_ANGLE)

    assert error < 1.0
