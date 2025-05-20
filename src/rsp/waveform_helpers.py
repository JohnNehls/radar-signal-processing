import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy import fft
from scipy.interpolate import interp1d
from scipy import signal

logger = logging.getLogger(__name__)

def zeropad_waveform(t, waveform, N_pad):
    """
    Zeropad waveform at the end of the pulse and modify time array accordingly.
    Args:
        t (1d array) : Time array
        waveform (1d array) : Signal array
        N_pad (int) : number of zero pad samples
    Return:
        noise : 1d array
    """
    assert type(N_pad) is int
    assert len(t) == len(waveform)
    waveform = np.append(waveform, np.zeros(N_pad))
    t = np.arange(waveform.size) * (t[1] - t[0]) + t[0]
    return t, waveform


def moving_average(waveform, N_elements):
    """
    Moving average of input waveform over a number of elements
    Args:
        waveform (1d array) : Signal array
        N_elements (int) : Number of elements to average over
    Return:
        ave : 1d array
    """
    assert type(N_elements) is int
    kernel = np.ones(N_elements) / N_elements
    ave = np.convolve(waveform, kernel, mode="same")
    return ave


def find_width(x, y, interp_max=5, interp_count=0, interp_scale=2):
    """
    Experimental/recreation: Find the width in x of the real signal y written for as a recursive algorithm.
    Args:
        x (array) : Independent variable
        y (array) : Dependent variable
    Return:
        pulsewidth, x_start, x_end : float, float, float
    """
    # Step 1: Find the maximum amplitude of the signal
    max_amplitude = np.max(y)

    # Step 2: Calculate the half maximum amplitude
    half_max_amplitude = max_amplitude / 2

    # Step 3: Find the points in the signal where the y is equal to the half maximum amplitude
    ind = np.where(np.isclose(y, half_max_amplitude, rtol=1e-2))[0]

    # base case
    if ind.size >= 2:
        t_start = x[ind[0]]
        t_end = x[ind[-1]]
        pulse_width = t_end - t_start
        return pulse_width, t_start, t_end

    # stop infinite recursion
    elif interp_count >= interp_max:
        logger.error("Error: cannot find width")
        return [np.nan, np.nan, np.nan]

    # recursive step
    else:
        logger.info(
            f"find_width: find_width is interpolating to 2x the sample rate {interp_count=}"
        )
        interp_func = interp1d(x, y, kind="linear")
        newx = np.linspace(x[0], x[-1], x.size * interp_scale)
        newy = interp_func(newx)
        return find_width(newx, newy, interp_max, interp_count + 1)


def plot_pulse_and_spectrum(t, mag, title=None, Npad=0, printBandwidth=True, spec_dec=False):
    """plotPulseAndSpectrum"""
    ## time domain ##
    fig, ax = plt.subplots(1, 2)
    if np.iscomplexobj(mag):
        ax[0].plot(t, np.abs(mag), "-o", label="magnitude")
        ax[0].plot(t, np.real(mag), "--o", label="real")
        ax[0].plot(t, np.imag(mag), "-.o", label="imag")
        ax[0].legend()
    else:
        ax[0].plot(t, mag, "-o")
        ax[0].set_ylim((min(0, mag.min() * 1.1), mag.max() * 1.1))

    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("baseband signal")
    ax[0].grid()

    ##  frequency domain ##
    # pad with zeros for greater freq resolution
    mag = np.append(mag, np.zeros(Npad))
    dt = t[1] - t[0]
    N = mag.size

    MAG = fft.fftshift(fft.fft(mag)) / N
    f = fft.fftshift(fft.fftfreq(N, dt))

    val = abs(MAG)
    val = val / val.max()
    if spec_dec:
        val = 20 * np.log10(val)
        ax[1].set_ylabel("baseband magnitude dB")
    else:
        ax[1].set_ylabel("baseband magnitude")
    ax[1].plot(f, val, "-")
    ax[1].set_xlabel("freqency [Hz]")

    ax[1].grid()

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if printBandwidth:
        print("\tbandwidth:")
        PW, f_start, f_end = find_width(f, abs(MAG))
        print(f"\t{PW=:.1e} {f_start=:.1e} {f_end=:.1e}")
        # PW_av, f_start_av, f_end_av = find_width(f, abs(moving_average(abs(MAG), 3)))
        # print(f"\t{PW_av=:.1e} {f_start_av=:.1e} {f_end_av=:.1e}")

    return fig, ax


def autocorrolate_waveform(waveform):
    """
    Create the autocorrelation of a waveform.
    Args:
        waveform (array) : real or complex waveform array
    Return:
        autoCor, index_shift : auto correlation and index shift arrays
    Notes:
        - index_shift is #TODO
    """
    Nwf = waveform.size
    Nfft = 2 * Nwf - 1  # add some padding
    WF = fft.fft(waveform, Nfft)
    autoCor = fft.ifft(WF * np.conj(WF))
    autoCor = fft.fftshift(autoCor)
    index_shift = np.arange(-(Nwf - 1), Nwf)

    return autoCor, index_shift


def plot_pulse_and_xcorrelation(t, mag, title=None, printWidth=True):
    """plotPulseAndautocorrelation"""
    dt = t[1] - t[0]

    fig, ax = plt.subplots(1, 2)
    if np.iscomplexobj(mag):
        ax[0].plot(t, np.abs(mag), "-o", label="magnitude")
        ax[0].plot(t, np.real(mag), "--o", label="real")
        ax[0].plot(t, np.imag(mag), "-.o", label="imag")
        ax[0].legend()

    else:
        ax[0].plot(t, mag, "-o")

    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("baseband signal")
    ax[0].grid()

    xcor, index_shift = autocorrolate_waveform(mag)
    time_shift = index_shift * dt
    val = abs(xcor)
    val = val / val.max()
    ax[1].plot(time_shift, val, "-o")
    ax[1].set_xlabel("time shift [s]")
    ax[1].set_ylabel("cross correlation mag")
    ax[1].grid()

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if printWidth:
        print("\txcor:")
        PW, f_start, f_end = find_width(time_shift, abs(xcor))
        print(f"\t{PW=:.1f} {f_start=:.1f} {f_end=:.1f}")
        PW_av, f_start_av, f_end_av = find_width(time_shift, moving_average(abs(xcor), 3))
        print(f"\t{PW_av=:.1f} {f_start_av=:.1f} {f_end_av=:.1f}")

    return fig, ax


def add_waveform_at_index(ar, waveform, index):
    """Add waveform to input array in place.
    Args:
        ar (array) : Independent array
        waveform (array) : real or complex waveform array
        index (int) : Index to place the waveform
    Return:
        ar : array
    """
    Nar = ar.size
    Nwv = waveform.size

    if index >= Nar:
        logger.info("add_waveform_at_index: wave form not added")
    elif index + Nwv >= Nar:
        ar[index:-1] = ar[index:-1] + waveform[: int(Nar - index - 1)]
        logger.info(
            f"add_waveform_at_index: add eclipsed waveform \n\t{index}\n\t{Nar}\n\t{Nwv}"
        )
    else:
        ar[index : index + Nwv] = ar[index : index + Nwv] + waveform
        logger.info(f"add_waveform_at_index: add waveform \n\t{index}\n\t{Nar}\n\t{Nwv}")
    return ar


def matchfilter_with_waveform(ar, waveform):
    """
    Create an array with the match filter of a waveform applied to an array.
    Args:
        ar (array) : Independent array
        waveform (array) : real or complex waveform array
    Return:
        index_shift, conv : index shift arrays and matched filter result
    #TODO describe what index_shift is
    """
    Nar = ar.size
    kernel = np.conj(waveform)[::-1]
    conv = signal.convolve(ar, kernel, mode="same", method="direct")
    if Nar % 2 == 0:
        index_shift = np.arange(-int(Nar / 2), int(Nar / 2))
    else:
        index_shift = np.arange(-int(Nar / 2), int(Nar / 2) + 1)

    return index_shift, conv
