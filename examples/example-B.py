#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:36:16 2021

@author: jiayingweng
"""
import FTIRE
X = generateX(100, 50, 0)
y, d = generateY(X, 1)
B = CV(X, y, d, 30, "ft")

