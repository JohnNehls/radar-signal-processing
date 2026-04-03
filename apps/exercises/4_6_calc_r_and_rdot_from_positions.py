#!/usr/bin/env python
"""Compute range and range-rate from 3D position and velocity vectors.

Given a radar and target each with a 3D position [x, y, z] and velocity
[vx, vy, vz], compute:
  - range: the Euclidean distance between them
  - range-rate: the rate of change of that distance (radial velocity)

Example: radar at origin, altitude 3048 m, moving east at 300 m/s;
         target at 5 km east, same altitude, moving west at 300 m/s.
"""

from rad_lab.geometry import range_and_rangerate

print("##########################")
print("3D radial calcs")
print("##########################")
print("Problem 6")

# range_and_rangerate(radar_pos, radar_vel, target_pos, target_vel)
result = range_and_rangerate([0, 0, 3048], [300, 0, 0], [5e3, 0, 3048], [-300, 0, 0])

print(result)
