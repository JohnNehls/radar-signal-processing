# rad-lab

A Python radar module for simulating pulse-Doppler returns and generating
range-Doppler maps (RDMs). Designed for radar engineers and students who want
to build intuition for how RDMs are formed, how waveforms affect resolution,
and how DRFM electronic attack techniques appear in the RDM.

## Quick start

```python
from rad_lab import rdm, Radar, Target, Return, barker_coded_waveform

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
    dwell_time=2e-3,
)

waveform = barker_coded_waveform(10e6, nchips=13)

return_list = [Return(target=Target(range=0.5e3, range_rate=1.0e3, rcs=1))]

rdm.gen(radar, waveform, return_list)
```

## Installation

```shell
pip install rad-lab
```

Or clone the repository for the full set of example apps:

```shell
git clone https://github.com/JohnNehls/rad-lab
pip install rad-lab
```
