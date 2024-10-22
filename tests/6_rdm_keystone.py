#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt

# Can make plotting non-blocking with an input flag
if sys.argv[-1].lower() == "--no-block":
    BLOCK = False
else:
    BLOCK = True


plt.show(block=BLOCK)

raise Exception("This funcionality/test is incomplete")
