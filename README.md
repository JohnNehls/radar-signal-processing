# rad-lab

[![CI](https://github.com/JohnNehls/rad-lab/actions/workflows/python-app.yml/badge.svg)](https://github.com/JohnNehls/rad-lab/actions/workflows/python-app.yml)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://johnnehls.github.io/rad-lab/)

A Python radar module for simulating pulse-Doppler returns and generating
range-Doppler maps (RDMs). Designed for radar engineers and students who want
to build intuition for how RDMs are formed, how waveforms affect resolution,
and how DRFM electronic attack techniques appear in the RDM.

## Installation

#### Install from PyPI (library only)
``` shell
pip install rad-lab
```

#### Clone for the full example apps
``` shell
git clone https://github.com/JohnNehls/rad-lab
pip install -e ./rad-lab
```

> A few exercises use LaTeX for plot labels — LaTeX must be installed for those to run.

## Usage

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

![image](docs/figs/rdm_readme_example.png)

Other available waveforms: `uncoded_waveform`, `random_coded_waveform`, `lfm_waveform`.
For additional examples see [apps/rdms](apps/rdms) and [apps/exercises](apps/exercises),
or the [API docs](https://johnnehls.github.io/rad-lab/).

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request.

To run the test suite:

```shell
python -m pytest tests/ -v
./apps/run_apps.sh   # runs all apps with a headless backend
```

## License

This project is licensed under the GPL-3.0 License - see
[LICENSE](LICENSE) for details.
