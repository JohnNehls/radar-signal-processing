import sys
import numpy as np
import matplotlib.pyplot as plt

from constants import PI, C

def plotRTM (r_axis, data, title):
    """Plot range-time matrix"""
    pulses = range(data.shape[1])
    fig, ax = plt.subplots(1,2)
    fig.suptitle(title)
    p = ax[0].pcolormesh(pulses, r_axis*1e-3, abs(data))
    ax[0].set_xlabel("pulse number")
    ax[0].set_ylabel("range [km]")
    ax[0].set_title("magnitude")
    fig.colorbar (p)
    ax[1].pcolormesh(pulses, r_axis*1e-3, np.angle(data))
    ax[1].set_xlabel("pulse number")
    ax[1].set_ylabel("range [km]")
    ax[1].set_title("phase")
    fig.tight_layout ()


def setZeroToSmallestNumber (array):
    smallest_float32= sys.float_info.min + sys.float_info.epsilon
    indxs = np.where(array==0)
    array[indxs] = smallest_float32


def plotRDM(rdot_axis, r_axis, data, title, cbarRange=30, volt2db=True):
    """Plot range-Doppler matrix"""
    data = abs(data)
    fig, ax = plt.subplots(1,1)
    fig.suptitle(title)
    if volt2db:
        setZeroToSmallestNumber(data)
        data = 20*np.log10(data)
    p = ax.pcolormesh (rdot_axis*1e-3, r_axis*1e-3, data)
    ax.set_xlabel("range rate [km/s]")
    ax.set_ylabel("range [km]")
    ax.set_title("magnitude squared")
    if cbarRange:
        p.set_clim((data.max() - cbarRange, data.max()))
    cbar = fig.colorbar(p)
    if volt2db:
        cbar.set_label("SNR [dB]")
    else:
        cbar.set_label("SNR")
    fig.tight_layout ()
