import numpy as np
import matplotlib.pyplot as plt


def a_f(theta, el_positions, weights=None):
    "elemetent positions are in terms of wavelength of light emmited"
    if weights is None:
        weights = np.ones(el_positions.shape)
    weighted_el = np.multiply(el_positions, weights)
    return np.sum(np.exp(1j * 2 * np.pi * np.sin(theta) * weighted_el))


def steering_vector(el_pos, theta):
    """Vandermonde Steering Vector
    Parameters
    el_pos : coordinates of array elements in units of wavelength
    theta : angle of steering in degrees
    """
    theta_rad = np.deg2rad(theta)
    return np.exp(-1j * 2 * np.pi * np.sin(theta_rad) * el_pos)


def linear_antenna_gain(Nel, dx, weights=None, Ntheta=10000, steer_angle=0, plot=False):
    """Create antena gain pattern for linear antenna"""
    L = (Nel - 1) * dx
    el_pos = np.linspace(-L / 2, L / 2, Nel)  # wavelengths

    if weights is None:
        weights = np.ones(Nel).T

    theta_grid = np.linspace(-np.pi / 2, np.pi / 2, Ntheta)

    # simple way of computing
    # af = np.array( [a_f(tmp, el_pos) for tmp in theta_grid] )

    steer_vec = steering_vector(el_pos, steer_angle)
    weights = weights * steer_vec

    # more efficent way of computing
    A = np.exp(1j * 2 * np.pi * np.outer(np.sin(theta_grid), el_pos))
    Af = A @ weights

    # The difference between the antenna's gain and the isotropic antenna's gain is  dBi
    af_dbi = 20 * np.log10(abs(Af))

    if plot:
        plt.plot(np.rad2deg(theta_grid), af_dbi)
        plt.ylim((-30, af_dbi.max() * 1.2))
        plt.xlabel(r"Angle $\theta$ [deg]")
        plt.ylabel("Gain [dBi]")
        plt.grid()

    return np.rad2deg(theta_grid), af_dbi
