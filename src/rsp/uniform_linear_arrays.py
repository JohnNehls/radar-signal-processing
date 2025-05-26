import numpy as np
import matplotlib.pyplot as plt
from . import constants as c
from scipy import interpolate


def steering_vector(el_pos, theta):
    """
    Vandermonde Steering Vector
    Args:
        el_pos : coordinates of array elements in units of wavelength
        theta : angle of steering in degrees
    Return:
        steering_vector
    """
    theta_rad = np.deg2rad(theta)
    return np.exp(-1j * 2 * np.pi * np.sin(theta_rad) * el_pos)


def linear_antenna_gain(el_pos, weight_vec=None, N_theta=10000, steer_angle=0, plot=False):
    """
    Antenna voltage gain pattern (complex) from element positions in terms of signal wavelength
    Args:
        el_pos : Coordinates of array elements in units of wavelength [wavelength]
        weight_vec : Array element weighting [unitless]
        N_theta : Number of theta points for grid (default 10000)
        steer_angle : Angle of maximum gain (default is 0)[rad]
        plot :  Plotting flag (default is False)
    Return:
        theta_vec, gain_vec : Theta grid and gain vector
    """
    if weight_vec is None:
        weight_vec = np.ones(len(el_pos)).T

    theta_grid = np.linspace(-np.pi / 2, np.pi / 2, N_theta)

    steer_vec = steering_vector(el_pos, steer_angle)
    weight_vec = weight_vec * steer_vec

    A = np.exp(1j * 2 * np.pi * np.outer(np.sin(theta_grid), el_pos))
    Af = A @ weight_vec

    # The difference between the antenna's gain and the isotropic antenna's gain is dBi
    af_dbi = 20 * np.log10(abs(Af))

    if plot:
        plt.figure()
        plt.plot(np.rad2deg(theta_grid), af_dbi)
        plt.ylim((-30, af_dbi.max() * 1.2))
        plt.xlabel(r"Angle $\theta$ [deg]")
        plt.ylabel("Gain [dBi]")
        plt.grid()

    return np.rad2deg(theta_grid), Af


def linear_antenna_gain_meters(el_pos, fc, weights=None, Ntheta=10000, steer_angle=0, plot=False):
    """
    Antenna voltage gain pattern (complex) from element positions [m] and signal frequency [Hz]
    Args:
        el_pos : Coordinates of array elements [m]
        fc : Center frequency of transmit/recieve [Hz]
        weight_vec : Array element weighting [unitless]
        N_theta : Number of theta points for grid (default 10000)
        steer_angle : Angle of maximum gain (default is 0)[rad]
        plot :  Plotting flag (default is False)
    Return:
        theta_vec, gain_vec : Theta grid and gain vector
    """
    wavelength = c.C / fc
    el_pos_per_wl = el_pos / wavelength
    return linear_antenna_gain(
        el_pos_per_wl, weight_vec=weights, N_theta=Ntheta, steer_angle=steer_angle, plot=plot
    )


def linear_antenna_gain_N_db(N_el, dx, weights=None, Ntheta=10000, steer_angle=0, plot=False):
    """
    Antenna voltage gain pattern (complex) in dBi from number of array elements and the elemnt spacing in terms of signal wavelength
    Args:
        N_el : Number of antenna array elements
        dx : Array spacing in wavelength [wavelength]
        weight_vec : Array element weighting [unitless]
        N_theta : Number of theta points for grid (default 10000)
        steer_angle : Angle of maximum gain (default is 0)[rad]
        plot :  Plotting flag (default is False)
    Return:
        theta_vec, gain_vec : Theta grid and gain vector
    """
    L = (N_el - 1) * dx
    el_pos = np.linspace(-L / 2, L / 2, N_el)  # wavelengths

    thetas, gain = linear_antenna_gain(
        el_pos, weight_vec=weights, N_theta=Ntheta, steer_angle=steer_angle, plot=plot
    )
    return thetas, 20 * np.log10(abs(gain))


def array_phase_center(position_ar, weight_ar):
    """
    Phase center location of an array form the positions and the weights.
    Args:
        position_ar : Coordinates of array elements #TODO [unit?]
        weight_ar : Array element weighting [unitless]
    Return:
        phase_center : float
    """
    assert len(position_ar) == len(weight_ar)
    # unsure if the weight should be abs value or not
    return np.sum(abs(weight_ar) * position_ar) / np.sum(weight_ar)


def apply_timeshift_due_to_element_position(signal_ar, fs, element_position, tgt_angle):
    """
    Apply time shift to signal due to element position [meters]
    Args:
        signal_ar (complex array): Time series of signal from a single antenna element [V]
        fs (float) : Sampling frequency [Hz]
        element_pos (float) : Array element's position relative to the phase center [m]
        tgt_angle (float) : Angle of the target relative to the array element [rad]
    Return:
        shift_signal_ar (complex array) : Time shifted signal
    """
    range_diff = element_position * np.sin(np.deg2rad(tgt_angle))
    time_shift = range_diff / c.C
    print(f"{time_shift=}")
    time_ar = np.arange(len(signal_ar)) / fs
    shifted_time = time_ar + time_shift
    interp_fun = interpolate.interp1d(time_ar, signal_ar, kind="cubic", fill_value="extrapolate")
    shifted_signal = interp_fun(shifted_time)

    return shifted_signal
