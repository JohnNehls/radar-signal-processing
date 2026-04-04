# rad-lab

[![CI](https://github.com/JohnNehls/radar-signal-processing/actions/workflows/python-app.yml/badge.svg)](https://github.com/JohnNehls/radar-signal-processing/actions/workflows/python-app.yml)

A Python radar module for simulating pulse-Doppler returns and generating
range-Doppler maps (RDMs). Designed for radar engineers and students who want
to build intuition for how RDMs are formed, how waveforms affect resolution,
and how DRFM electronic attack techniques appear in the RDM.

## Modules

-   [rdm](src/rad_lab/rdm.py) — range-Doppler map generation
-   [pulse_doppler_radar](src/rad_lab/pulse_doppler_radar.py) — radar system parameter model
-   [waveform](src/rad_lab/waveform.py) — uncoded, Barker, random-coded, and LFM pulse generation
-   [returns](src/rad_lab/returns.py) — skin return and DRFM jammer return models
-   [range_equation](src/rad_lab/range_equation.py) — radar and one-way link range equations
-   [uniform_linear_arrays](src/rad_lab/uniform_linear_arrays.py) — ULA gain patterns and steering vectors
-   [monopulse](src/rad_lab/monopulse.py) — amplitude monopulse angle estimation
-   [vbm](src/rad_lab/vbm.py) — velocity bin masking EA slow-time modulation functions
-   [geometry](src/rad_lab/geometry.py) — range and range-rate from geometry
-   [noise](src/rad_lab/noise.py) — complex Gaussian noise generation
-   [utilities](src/rad_lab/utilities.py) — unit conversions and signal utilities

## Installation

### Requirements

-   Python >= 3.11
-   Python packages listed in [pyproject.toml](pyproject.toml)
-   A few exercises use LaTeX for plot labels — LaTeX must be installed for
    those to run

### Installation options

#### (Option 1) Clone the repository with all the apps and docs
``` shell
git clone https://github.com/JohnNehls/rad-lab
pip install rad-lab
```

#### (Option 2) Install from PyPI (library only, no apps)
``` shell
pip install rad-lab
```

## Usage

### RDM generator

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

For additional examples including DRFM jammer returns and VBM, see
[apps/rdms](apps/rdms). [kitchen_sink.py](apps/rdms/kitchen_sink.py) shows
all waveform and return options.

### Everything else

For examples of the other module functions, see the
[exercises](apps/exercises).

## Testing

-   To run the pytests:

``` shell
python -m pytest tests/ -v
```

-   To run all apps and check for errors:

``` shell
./apps/run_apps.sh
```

This script runs all Python files in `apps/exercises/`, `apps/rdms/`, and
`apps/studies/` with the `Agg` matplotlib backend so no display is required.
Files ending in `_no_test.py` are skipped. The script exits with a non-zero
status if any file fails.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull
request.

## License

This project is licensed under the GPL-3.0 License - see
[LICENSE](LICENSE) for details.
