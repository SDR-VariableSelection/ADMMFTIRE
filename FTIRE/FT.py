#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 21:53:32 2020

@author: jiayingweng
"""

import numpy as np
from scipy import stats
import scipy.linalg as la
import FTIRE

__all__ = ['FT']
#%%

class FT:
    """FT method to apply for X and Y"""
    def __init__(self, X, y, d = 1, m = 30, weighted = False, W = None, spX = False):
        self.d = d
        self.m = m
        self.X = np.array(X)
        self.y = np.array(y)
        self.M = None
        self.B = None
        self.values = None
        self.pval = None
        self.test = None
        self.weighted = weighted
        self.W = W
        self.spX = spX

    
    def __str__(self):
        return "This is FT method (standardized f(y)) applied for {} dimension reduction with m={} and weight={}.".format(self.d, self.m, self.weighted)
    
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
    
    
    def kernel(self, weighted=None):
        if weighted is None:
            weighted = self.weighted
        n, p = self.X.shape
        self.y = self.y.reshape(n,-1)
        q = self.y.shape[1]
        X0 = self.X - np.mean(self.X, axis = 0)
        if self.W is None:
            sig = 0.1 * np.pi ** 2/ np.median(la.norm(self.y, axis = 1) **2)
            self.W = np.random.rand(q, self.m) * sig
        # Mxy = np.zeros((p, 2*self.m))
        My = np.zeros((n, 2*self.m))
        for i in range(self.m):
            w = self.W[:,i]
            if q > 1:
                w = w/( np.linalg.norm(self.W[:,i]) * np.ones((1,q)) )
            yw =  self.y @ w.T #* 2 * np.pi      
            cosy = np.array(np.cos(yw))
            cosy = (cosy - np.mean(cosy, axis = 0))/np.std(cosy, axis = 0)
            My[:, 2*i] = np.array(cosy).reshape(-1)
            siny = np.sin(yw)
            siny = (siny - np.mean(siny, axis = 0))/np.std(siny, axis = 0)
            My[:, 2*i + 1] = np.array(siny).reshape(-1)
        
        sigcov = np.cov(np.concatenate((X0, My), axis = 1),rowvar=False)
        sigxf = sigcov[0:p, p:]
        sigff = sigcov[p:, p:]
        
        if weighted:
            ivcovy = self.matpower(cov = sigff, a = -1)
            self.M = sigxf @ ivcovy @ sigxf.T
        else:
            self.M = sigxf @ sigxf.T
        
        return sigxf
    
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
            self.pval[r] = 1 - stats.chi2.cdf(self.test[r], (p-r)*(2*self.m-r))
        return covX

    
def matpower(cov, a, err=1e-3):
    value, vector = la.eigh(cov)
    indv = value > err
    trcov = vector[:,indv] @ np.diag(np.power(value[indv], a)) @ vector[:,indv].T
    return trcov
    
