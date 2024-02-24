import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy import interpolate
from scipy.interpolate import interp1d

def moving_average(waveform, elements):
    kernel = np.ones(elements)/elements
    av = np.convolve(waveform, kernel, mode='same')
    return av

def find_width(x, y,  interp_max=5, interp_count=0,interp_scale=2, debug=False):
    """find the width in x of the real signal y written for recreation"""
    # Step 1: Find the maximum amplitude of the signal
    max_amplitude = np.max(y)

    # Step 2: Calculate the half maximum amplitude
    half_max_amplitude = max_amplitude / 2

    # Step 3: Find the points in the signal where the y is equal to the half maximum amplitude
    ind = np.where(np.isclose(y, half_max_amplitude, rtol=1e-2))[0]

    # base case
    if ind.size  >= 2:
        t_start = x[ind[0]]
        t_end = x[ind[-1]]
        pulse_width = t_end - t_start
        return pulse_width, t_start, t_end

    # stop infinite recursion
    elif interp_count >= interp_max:
        if debug:
            print("Error: cannot find width")
        return [np.nan, np.nan, np.nan]

    # recursive step
    else:
        if debug:
            print(f"find_width is interpolating to 2x the sample rate {interp_count=}")
        interp_func = interp1d(x, y, kind='linear')
        newx = np.linspace(x[0],x[-1],x.size*interp_scale)
        newy = interp_func(newx)
        return find_width(newx, newy, interp_max, interp_count+1)


## problem 1 ###########################################
def plotPulseAndSpectrum(t, mag, title=None, printBandwidth=True):
    dt = t[1] - t[0]
    T = t[-1] - t[0]
    N = mag.size

    fig, ax = plt.subplots(1,2)
    if np.iscomplexobj(mag):
        ax[0].plot(t, np.abs(mag),'-',label='magnitude')
        ax[0].plot(t, np.real(mag),'--',label='real')
        ax[0].plot(t, np.imag(mag),'-.',label='imag')

        ax[0].legend()
    else:
        ax[0].plot(t, mag)

    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("baseband signal")
    ax[0].grid()

    sampleRate = 1/dt
    MAG = fft.fftshift( fft.fft(mag) )/N
    f = np.linspace(-sampleRate/2,sampleRate/2, N)
    val = abs(MAG)
    val = val/val.max()
    ax[1].plot(f, val)
    ax[1].set_xlabel("freqency [Hz]")
    ax[1].set_ylabel("baseband magnitude")
    ax[1].grid()

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if printBandwidth:
        print("bandwidth:")
        PW, f_start, f_end = find_width(f, abs(MAG))
        print(f"\t{PW=:.1f} {f_start=:.1f} {f_end=:.1f}")
        PW_av, f_start_av, f_end_av = find_width(f, abs(moving_average(abs(MAG), 3)))
        print(f"\t{PW_av=:.1f} {f_start_av=:.1f} {f_end_av=:.1f}")

    return fig, ax

def autocorrolate_waveform(waveform):
    Nwf = waveform.size
    Nfft = 2*Nwf - 1  # add some padding
    WF = fft.fft(waveform, Nfft)
    val = fft.ifft(WF*np.conj(WF))
    val = fft.fftshift(val)
    index_shift = np.arange(-(Nwf-1), Nwf)

    return val, index_shift

def plotPulseAndCrossCorrelation(t, mag, title=None, printWidth=True):
    dt = t[1] - t[0]
    T = t[-1] - t[0]
    N = mag.size

    fig, ax = plt.subplots(1,2)
    if np.iscomplexobj(mag):
        ax[0].plot(t, np.real(mag),'--o',label='real',linewidth=0.1)
        ax[0].plot(t, np.imag(mag),'-.g',label='imag',linewidth=0.1)
        ax[0].plot(t, np.abs(mag),'-b',label='magnitude',linewidth=2)
        ax[0].legend()
    else:
        ax[0].plot(t, mag)

    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("baseband signal")
    ax[0].grid()

    xcor, index_shift = autocorrolate_waveform(mag)
    time_shift = index_shift*dt
    val = abs(xcor)
    val = val/val.max()
    ax[1].plot(time_shift, val)
    ax[1].set_xlabel("time shift")
    ax[1].set_ylabel("cross correlation mag")
    ax[1].grid()

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if printWidth:
        print("xcor:")
        PW, f_start, f_end = find_width(time_shift, abs(xcor))
        print(f"\t{PW=:.1f} {f_start=:.1f} {f_end=:.1f}")
        PW_av, f_start_av, f_end_av = find_width(time_shift, moving_average(abs(xcor), 3))
        print(f"\t{PW_av=:.1f} {f_start_av=:.1f} {f_end_av=:.1f}")

    return fig, ax
