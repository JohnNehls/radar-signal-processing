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

-   Python \>= 3.10
-   Python packages listed in [pyproject.toml](pyproject.toml)
-   A few of the tests utilize LaTeX, thus it may need to be installed
    in order for them to run

## Usage

-   **RDM generator**
    -   For examples of basic usage, see the [examples](examples).
        -   [kitchen_sink_.py](examples/rdms/kitchen_sink.py) is a
            script with all waveform and return options written out
-   **Everything else**
    -   For example of the other functions of the project, see
        [tests](examples/tests).

## Testing RSP

-   To run each python script in [tests](examples/tests), [examples](examples),
    and [studies](examples/studies) to test for errors, run the following:

``` shell
./examples/tests/run_all_tests.sh
```

### Known Test Failures

1.  [6_0_rdm_keystone.py](examples/tests/6_0_rdm_keystone.py)

    -   Not yet implemented

2.  [7_4_linear_arrays_ifft_array_factor.py](examples/tests/7_4_linear_arrays_ifft_array_factor.py)

    -   Not yet implemented

## Contributing

Contributions are welcome. Please fork the repository and submit a pull
request.

## License

This project is licensed under the GPL-3.0 License - see
[LICENSE](LICENSE) for details.
