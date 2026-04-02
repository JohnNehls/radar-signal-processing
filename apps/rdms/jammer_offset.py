import matplotlib.pyplot as plt
from rsp import rdm, Radar, Target, Return, EaPlatform, lfm_waveform

radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=20e6,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=3e-3,
)

waveform = lfm_waveform(bw=10e6, T=1.0e-6, chirp_up_down=1)

return_list = [
    Return(target=Target(range=3.5e3, range_rate=1e3, rcs=10),
           platform=EaPlatform(
               tx_power=0.5, tx_gain=10 ** (3 / 10), total_losses=10 ** (5 / 10),
               rdot_delta=2.0e3, rdot_offset=-1e3, range_offset=-0.2e3, delay=0))]

rdm.gen(radar, waveform, return_list)
plt.show()
