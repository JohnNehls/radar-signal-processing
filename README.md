# Radar Signal Processing

Range Doppler map (RDM) and general radar signal processing module.

## Description

This Python module provides tools for creating range Doppler maps and a
variety of functions useful for radar signal processing. It is designed
to be a simple tool that is useful for gaining intuition on how RDMs are
made.

## Libraries

-   [RDM generation](src/rsp/rdm.py)
-   [waveform generation](src/rsp/waveform.py)
-   [pulse doppler radar](src/rsp/pulse_doppler_radar.py)
-   [radar range equation](src/rsp/range_equation.py)
-   [uniform linear array antennas](src/rsp/uniform_linear_arrays.py)
-   [monopulse angle estimation](src/rsp/monopulse.py)
-   [geometry](src/rsp/geometry.py)
-   [velocity bin masking (VBM)](src/rsp/vbm.py)
-   [noise](src/rsp/noise.py)
-   [utilities](src/rsp/utilities.py)

## Installation

To install the module, clone this repository and install with pip:

``` shell
git clone https://github.com/JohnNehls/radar-signal-processing
pip install radar-signal-processing/
```

### requirements

-   Python \>= 3.11
-   Python packages listed in [pyproject.toml](pyproject.toml)
-   A few of the exercises utilize LaTeX for plot labels, thus it may
    need to be installed in order for them to run

## Usage

-   **RDM generator**

    ```python
    from rsp import rdm
    from rsp.pulse_doppler_radar import Radar
    from rsp.waveform import barker_waveform
    from rsp.returns import Target, Return

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
        dwell_time=2e-3,
    )

    waveform = barker_waveform(10e6, nchips=13)

    return_list = [Return(target=Target(range=0.5e3, rangeRate=1.0e3, rcs=1))]

    rdm.gen(radar, waveform, return_list)
    ```

    ![image](docs/figs/rdm_readme_example.png)


    -   For additional examples, see [apps/rdms](apps/rdms).
        -   [kitchen_sink.py](apps/rdms/kitchen_sink.py) is a
            script with all waveform and return options written out

-   **Everything else**
    -   For examples of the other functions of the project, see
        [exercises](apps/exercises).

## Testing RSP
-   To run each of the pytests, run the following:

``` shell
python -m pytest tests/ -v
```

-   To ensure the main applications in [apps](apps/) run without errors and check for qualitative errors in the rdms, run the following:

    ``` shell
    ./apps/run_apps.sh
    ```

    This script runs all Python files in `apps/exercises/`, `apps/rdms/`, and
    `apps/studies/` with the `Agg` matplotlib backend so no display is required.
    Any file whose name ends with `_no_test.py` is skipped -- usually due to the exercise
    not being finished. The script exits with a non-zero status if any file fails.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull
request.

## License

This project is licensed under the GPL-3.0 License - see
[LICENSE](LICENSE) for details.
