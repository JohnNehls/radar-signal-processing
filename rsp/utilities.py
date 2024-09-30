#!/usr/bin/env python

import numpy as np

from .constants import PI, C, K_BOLTZ

def phase_negpi_pospi(phase: list):
    """return phase in [-pi, pi)"""

    if not hasattr(phase, '__iter__'):
        phase = [phase]

    phase = np.array(phase)
    phase = phase%(2*PI)
    # phase = phase.tolist()
    for i, p in enumerate(phase):
        if p >= PI:
            phase[i] = p - 2*PI
    return phase

def phase_zero_twopi(phase: list):
    """return phase in [-pi, pi)"""
    phase = np.array(phase)
    phase = phase%(2*PI)
    return phase

def main():
    ## test phase branch aliases ################################
    phase_ar = np.linspace(-3*PI, 5*PI, 100)
    phase_ar_negpi_pospi = phase_negpi_pospi(phase_ar)
    plt.figure()
    plt.title("[-pi,pi) phase branch test")
    plt.plot(phase_ar, phase_ar_negpi_pospi, 'o')
    plt.xlabel("input phase")
    plt.ylabel("output phase")
    plt.grid()
    print("check the that -pi is included and pi is not" )

    phase_ar = np.linspace(-4*PI, 4*PI, 100)
    phase_ar_zero_twopi = phase_zero_twopi(phase_ar)
    plt.figure()
    plt.title("[0, 2pi) phase branch test")
    plt.plot(phase_ar, phase_ar_zero_twopi, 'o')
    plt.xlabel("input phase")
    plt.ylabel("output phase")
    plt.grid()
    print("check the that 0 is included and 2pi is not" )

    plt.show()

if __name__ == "__main__":
    main()
