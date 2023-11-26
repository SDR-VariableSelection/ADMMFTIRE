# Description

This is a ADMM-FTIRE package for the paper "Fourier transform sparse inverse regression estimators for sufficient variable selection".	

# Installation

pip install ADMMFTIRE==0.0.3

# Example

import FTIRE

from FTIRE import *

X = genxy.generateX(100, 50, 0)

y, d, q, b = genxy.generateY(X, 1)

B = ftire.CV(X, y, d, 30, "ft")

# Citation
```
@article{weng2022fourier,
  title={Fourier transform sparse inverse regression estimators for sufficient variable selection},
  author={Weng, Jiaying},
  journal={Computational Statistics \& Data Analysis},
  volume={168},
  pages={107380},
  year={2022},
  publisher={Elsevier}
}
```
