import matplotlib.pyplot as plt
from rsp import rdm
from rsp.pulse_doppler_radar import Radar
from rsp.waveform import lfm_waveform
from rsp.returns import Target, Return, EaPlatform

radar = Radar(
    fcar=10e9,
    txPower=1e3,
    txGain=10 ** (30 / 10),
    rxGain=10 ** (30 / 10),
    opTemp=290,
    sampRate=20e6,
    noiseFactor=10 ** (8 / 10),
    totalLosses=10 ** (8 / 10),
    PRF=200e3,
    dwell_time=3e-3,
)

waveform = lfm_waveform(bw=10e6, T=1.0e-6, chirpUpDown=1)

return_list = [
    Return(target=Target(range=3.5e3, rangeRate=1e3, rcs=10),
           platform=EaPlatform(
               txPower=0.5, txGain=10 ** (3 / 10), totalLosses=10 ** (5 / 10),
               rdot_delta=2.0e3, rdot_offset=-1e3, range_offset=-0.2e3, delay=0))]

rdm.gen(radar, waveform, return_list)
plt.show()
