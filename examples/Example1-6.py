#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples
"""

import numpy as np
import scipy.linalg as la
import time

from FTIRE.ftire import CV, projloss
from FTIRE.genxy import generateX, generateY

# seedid = int(sys.argv[1])
seedid = 125
np.random.seed(seedid)
t0 = time.time()
#%%
n = 100
p = 1000
covstr = 0 ## 0-3
M = 1 ## Example 1,2,3,4,10,11
method = "ft" ## "ft" or "sir

X = generateX(n, p, covstr)
y, d, q, b = generateY(X, M)
m = 5 if method == 'sir' else 30
B, covxx, lambcv, cverr = CV(X, y, d, m, method)

loss = np.zeros(5)
loss[0] = seedid
loss[1] = projloss(B, b)

## loss and variable selection
S = np.nonzero(la.norm(b, axis = 1))[0]
Sh = np.nonzero(la.norm(B, axis = 1))[0]
# All = set(range(p))
S = set(S) 
# Sc = All - S
loss[2] = len(S.intersection(Sh)) ## True Positive
loss[3] = len(Sh) - loss[2] #len(Sc.intersection(Sh)) ## False Positive
loss[4] = (loss[2] == len(Sh)) + 0
print(np.round(loss, 3))

t1 = time.time()
print('elapsed time is', t1 - t0)
Sh = set(Sh)
## save file
# t = time.strftime("%y%m%d")
fname = f'admm_n{n}_p{p}_M{M}_SDR{method}_Cov{covstr}.txt'
print(fname)
f=open(fname,'a')
np.savetxt(f, loss, fmt='%1.4f', newline=" ")
f.write("\n")
f.close()

