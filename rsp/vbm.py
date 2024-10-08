import numpy as np
from numpy.linalg import norm
from . import constants as c
from .noise import band_limited_complex_noise, guassian_complex_noise
from .waveform import lfm_pulse

# Achieve Velocity Bin Masking (VBM) by adding pahse in slow time #########################
# - want to add phase so wvfm will sill pass radar's match filter


def calc_f_delta(fcar, rdot_delta):
    """ "convert rdot_delta to a frequency delta"""
    return 2 * fcar / c.C * rdot_delta


def print_noise_stats(slowtime_noise):
    """Print some noise stats"""
    print("\nBand Noise:")
    print(f"\t{norm(slowtime_noise)=}")
    print(f"\t{slowtime_noise.size=}")
    print(f"\t{np.mean(abs(slowtime_noise))=}")
    print(f"\t{np.var(abs(slowtime_noise))=}")
    print(f"\t{np.mean(np.angle(slowtime_noise))=}")
    print(f"\t{np.sqrt(np.var(np.angle(slowtime_noise)))=}")


####################################################################################################
### Start: noise techniques to achieve VBM in order of complexity ###
####################################################################################################
def _random_phase(Npulses):
    """ "Method 0 : random phase in all frequencies"""
    rand_phase = 2 * c.PI * np.random.rand(Npulses)
    return np.exp(1j * rand_phase)


def _uniform_bandwidth_phase(Npulses, f_delta, PRF):
    """ "Method 1 : random phase in a band width"""
    # - does not require assumption on processing interval
    # - dirty result if each element is made magnitude = 1
    # - un-normalized (normalized over interval) only makes sense if possible on hardware
    return band_limited_complex_noise(-f_delta / 2, +f_delta / 2, Npulses, PRF, normalize=True)


def _gaussian_bandwidth_phase(Npulses, f_delta, PRF):
    """ "Method 1 : random phase in a band width"""
    # - does not require assumption on processing interval
    # - dirty result if each element is made magnitude = 1
    # - un-normalized (normalized over interval) only makes sense if possible on hardware
    return guassian_complex_noise(0, f_delta / 2, 1, Npulses, PRF, normalize=True)


def _gaussian_bandwidth_phase_normalized(Npulses, f_delta, PRF):
    """ "Method 1.5 : random phase normalized over a period-- WIP"""
    # - A way to make the random noise cleaner is to normalize over a an interval
    # - use with un-normalized noise
    # - requires knowledge of number of pulses? (maybe)
    slowtime_noise = guassian_complex_noise(0, f_delta / 2, 1, Npulses, PRF, normalize=False)
    slowtime_noise = slowtime_noise / norm(slowtime_noise) * np.sqrt(slowtime_noise.size)
    return slowtime_noise


def _lfm_phase(Npulses, f_delta, PRF):
    """ "Method 2 : use LFM in slow time"""
    _, slowtime_noise = lfm_pulse(PRF, f_delta, Npulses / PRF, 1, normalize=False)
    return slowtime_noise


####################################################################################################
### End: noise techniques to achieve VBM in order of complexity ###
####################################################################################################


def slowtime_noise(Npulses, fcar, rdot_delta, PRF, noiseFun=_lfm_phase, debug=False):
    """Create noise in slowtime for VBM
    noiseFun choices: random_VBM, uniform_bandwidth_VMB, gaussian_bandwidth_VBM, gaussian_bandwidth_amp_VBM, lfm_VBM"""
    f_delta = calc_f_delta(fcar, rdot_delta)
    slowtime_noise = noiseFun(Npulses, f_delta, PRF)

    if debug:
        print_noise_stats(slowtime_noise)

    return slowtime_noise
