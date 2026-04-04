"""Physical and system constants used throughout rad-lab.

Defines the speed of light, Boltzmann's constant, a standard radar load
impedance, and other fixed values referenced by the range equation, noise,
and waveform modules.
"""

import math

PI = math.pi
C = 3e8  # m/s
K_BOLTZ = 1.38064852e-23  # m^2 kg / s^2 K
RADAR_LOAD = 500  # Ohms
