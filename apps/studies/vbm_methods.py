#!/usr/bin/env python

import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, EaPlatform, Return, uncoded_waveform
import rad_lab.vbm as vbm

################################################################################
# Display each of the VBM noise methods in order of complexity
################################################################################

bw = 5e6

radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=500e3,
    dwell_time=5e-3,
)

waveform = uncoded_waveform(bw)

vbm_name_function_dict = {
    "random phase VBM": vbm._random_phase,
    "uniform bandwidth phase VBM": vbm._uniform_bandwidth_phase,
    "gaussian bandwidth phase VBM": vbm._gaussian_bandwidth_phase,
    "uniform bandwidth phase normalized VBM": vbm._gaussian_bandwidth_phase_normalized,
    "LFM phase VBM": vbm._lfm_phase,
}

rdot_delta = 1.0e3
for name, func in vbm_name_function_dict.items():
    jammer_return = Return(
        target=Target(range=0.2e3, range_rate=0.0e3),
        platform=EaPlatform(
            tx_power=1.0e3,
            tx_gain=10 ** (5 / 10),
            total_losses=10 ** (3 / 10),
            rdot_delta=rdot_delta,
            rdot_offset=0.0e3,
            vbm_noise_function=func,
        ),
    )
    rdm.gen(radar, waveform, [jammer_return], debug=False)
    ax = plt.gca()
    ax.set_title(name)

print(
    f"Note:LFM phase VBM is the only method to match the perscribed {rdot_delta} [m/s] VBM width."
)
plt.show()
