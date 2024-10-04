#!/usr/bin/env python

from rsp.rdm_helpers import calc_range_and_rangeRate

print("##########################")
print("3D radial calcs")
print("##########################")
print("PROBLEM 6")
result = calc_range_and_rangeRate([0, 0, 3048], [300, 0, 0], [5e3, 0, 3048], [-300, 0, 0])

print(result)
