# Description

This is a ADMM-FTIRE package for the submitted paper "An Iterative ADMM Algorithm in Ultrahigh Dimensional Sufficient Dimension Reduction".	

# Installation

pip install ADMMFTIRE==0.0.2

# Example

import FTIRE

from FTIRE import *

X = genxy.generateX(100, 50, 0)

y, d, q, b = genxy.generateY(X, 1)

B = ftire.CV(X, y, d, 30, "ft")
