#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:36:16 2021

@author: jiayingweng
"""
import FTIRE
from FTIRE import *
X = genxy.generateX(100, 50, 0)
y, d, q, b = genxy.generateY(X, 1)
B = ftire.CV(X, y, d, 30, "ft")

