#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:03:10 2020

@author: jiayingweng
"""

import numpy as np
from scipy import stats
import scipy.linalg as la
import FTIRE

__all__ = ['SIR']

class SIR:
    """SIR method to apply for X and Y"""
    def __init__(self, X, y, d = 1, n_slice = 10, spX = False):
        self.d = d
        self.n_slice = n_slice
        self.X = np.array(X)
        self.y = np.array(y)
        self.M = None
        self.B = None
        self.values = None
        self.pval = None
        self.test = None
        self.spX = spX
    
    def __str__(self):
        return "This is SIR method applied for {} dimension reduction with {} slices.".format(self.d, self.n_slice)
    
    def dr(self, d = None):
        if d is None:
            d = self.d
        n, p = self.X.shape
        ## kernel matrix
        self.kernel()
        ## covariance matrix
        # n = self.X.shape[0]
        covX = FTIRE.SpCov.spcovCV(self.X) if self.spX else np.cov(self.X.T)
        ## SVD
        if p == 1:
            self.B = np.ones((p,d))
        elif n < p:
            self.B = np.zeros((p,d))
        else:
            self.values, vectors = la.eigh(self.M, covX)
            decsort = np.argsort(self.values)[::-1]
            self.B = vectors[:,decsort[range(d)]]
        return covX
    
    def upd(self):
        """ update dimension d """
        self.testd()
        self.d = next((i for i, j in enumerate(self.pval > 0.05) if j), None)
        if self.d is None:
            self.d = self.X.shape[1]
        self.dr(self.d)
        return self.B 
    
    def transformX(self):
        self.dr(self.d)
        newX = self.X @ self.B
        return newX
    
    def kernel(self):
        n, p = self.X.shape
        X0 = self.X - np.mean(self.X, axis = 0)
        # slice y into n_slice
        YI = np.argsort(self.y.reshape(-1))
        split = np.array_split(YI, self.n_slice)
        
        ## kernel matrix
        self.M = np.zeros((p, p))
        Ups = np.zeros((p, self.n_slice))
        for i in range(self.n_slice):
            if p == 1:
                tt = X0[split[i]].reshape(-1, p)
            else:
                tt = X0[split[i],].reshape(-1, p)   
            Xh = np.mean(tt, axis = 0).reshape(p, 1)
            Ups[:,i] = np.sqrt(len(split[i])/n) * Xh.reshape(-1)
            self.M = self.M + Xh @ Xh.T * len(split[i])/n
        return Ups
    
    
    def testd(self):
        n, p = self.X.shape
        ## kernel matrix
        self.kernel()
        ## covariance matrix
        covX = FTIRE.SpCov.spcovCV(self.X) if self.spX else np.cov(self.X.T)
        ## SVD
        self.values, vectors = la.eigh(self.M, covX)
        ## Sequential Tests
        decsort = np.argsort(np.abs(self.values))[::-1]
        self.pval = self.test = np.zeros(p)
        for r in range(p):
            self.test[r] = n * np.sum(self.values[decsort[r:p]])
            self.pval[r] = 1 - stats.chi2.cdf(self.test[r], (p-r)*(self.n_slice-1-r))
        return covX

    
    
    
    
