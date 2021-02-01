# Description

This is a ADMM-FTIRE package for the submitted paper "Sufficient Dimension Reduction in Ultrahigh Dimension with Fourier Transform".	

# Installation

pip install ADMMFTIRE

# Example

import FTIRE

from FTIRE import *

X = genxy.generateX(100, 50, 0)

y, d, q, b = genxy.generateY(X, 1)

B = ftire.CV(X, y, d, 30, "ft")
