import numpy as np
import matplotlib.pyplot as plt
from . import constants as c


def steering_vector(el_pos, theta):
    """Vandermonde Steering Vector
    Parameters
    el_pos : coordinates of array elements in units of wavelength
    theta : angle of steering in degrees
    """
    theta_rad = np.deg2rad(theta)
    return np.exp(-1j * 2 * np.pi * np.sin(theta_rad) * el_pos)


def linear_antenna_gain(el_pos, weights=None, Ntheta=10000, steer_angle=0, plot=False):
    """Antenna voltage gain pattern (complex) from element positions in terms of signal wavelength"""

    if weights is None:
        weights = np.ones(len(el_pos)).T

    theta_grid = np.linspace(-np.pi / 2, np.pi / 2, Ntheta)

    steer_vec = steering_vector(el_pos, steer_angle)
    weights = weights * steer_vec

    A = np.exp(1j * 2 * np.pi * np.outer(np.sin(theta_grid), el_pos))
    Af = A @ weights

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
    """Antenna gain pattern from element positions [m] and signal frequency [Hz]"""
    wavelength = c.C / fc
    el_pos_per_wl = el_pos / wavelength
    return linear_antenna_gain(
        el_pos_per_wl, weights=weights, Ntheta=Ntheta, steer_angle=steer_angle, plot=plot
    )


def linear_antenna_gain_N_db(Nel, dx, weights=None, Ntheta=10000, steer_angle=0, plot=False):
    """Antenna gain pattern in dBi from number of array elements and the elemnt spacing in terms of signal wavelength"""
    L = (Nel - 1) * dx
    el_pos = np.linspace(-L / 2, L / 2, Nel)  # wavelengths

    thetas, gain = linear_antenna_gain(
        el_pos, weights=weights, Ntheta=Ntheta, steer_angle=steer_angle, plot=plot
    )
    return thetas, 20 * np.log10(abs(gain))


def array_phase_center(position_ar, weight_ar):
    """Phase center location of an array form the positions and the weights."""
    assert len(position_ar) == len(weight_ar)
    # unsure if the weight should be abs value or not
    return np.sum(abs(weight_ar) * position_ar) / np.sum(weight_ar)
