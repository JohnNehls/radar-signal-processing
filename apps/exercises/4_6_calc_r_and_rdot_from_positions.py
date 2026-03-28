#!/usr/bin/env python

from rsp.geometry import range_and_rangerate

print("##########################")
print("3D radial calcs")
print("##########################")
print("Problem 6")
result = range_and_rangerate([0, 0, 3048], [300, 0, 0], [5e3, 0, 3048], [-300, 0, 0])

print(result)
