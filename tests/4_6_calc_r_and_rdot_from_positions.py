#!/usr/bin/env python

import numpy as np
import sys

sys.path.append("..")

from rf_datacube import R_pf_tgt

# constants
C = 3e8
PI = np.pi

print("##########################")
print("3D radial calcs")
print("##########################")
print("PROBLEM 6")
R_pf_tgt([0, 0, 3048], [300,0,0], [5e3,0,3048], [-300,0,0])
