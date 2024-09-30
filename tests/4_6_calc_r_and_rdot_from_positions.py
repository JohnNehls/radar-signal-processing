#!/usr/bin/env python

from rsp.rf_datacube import R_pf_tgt

print("##########################")
print("3D radial calcs")
print("##########################")
print("PROBLEM 6")
result = R_pf_tgt([0, 0, 3048], [300,0,0], [5e3,0,3048], [-300,0,0])

print(result)
