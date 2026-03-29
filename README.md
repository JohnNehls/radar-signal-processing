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
-   [pulse doppler](src/rsp/pulse_doppler_radar.py)
-   [radar range equation](src/rsp/range_equation.py)
-   [uniform linear array antennas](src/rsp/uniform_linear_arrays.py)
-   [noise](src/rsp/noise.py)

## Installation

To install the module, clone this repository and install with pip:

``` shell
git clone https://github.com/JohnNehls/radar-signal-processing
pip install radar-signal-processing/
```

### requirements

-   Python \>= 3.11
-   Python packages listed in [pyproject.toml](pyproject.toml)
-   A few of the tests utilize LaTeX, thus it may need to be installed
    in order for them to run

## Usage

-   **RDM generator**
    -   For examples of basic usage, see the [apps](apps).
        -   [kitchen_sink_.py](apps/rdms/kitchen_sink.py) is a
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
    Any file whose name ends with `_no_test.py` is skipped-- usually due to the exercise
	not being finished. The script exits with a non-zero status if any file fails.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull
request.

## License

This project is licensed under the GPL-3.0 License - see
[LICENSE](LICENSE) for details.
