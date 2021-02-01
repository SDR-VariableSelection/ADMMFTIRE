#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 01:24:00 2020

@author: jiayingweng
"""

import numpy as np
import scipy.linalg as la
from sklearn.model_selection import train_test_split

__all__ = ['spcov', 'spcovCV']

# covxx = np.cov(X.T)
def spcov(covxx, lam, standard):    
    # covxx = np.cov(X.T) * (n-1)/n
    # lam = 0.1
    # standard = True    
    if standard:
        covdiag = np.diag(covxx)
        dhat = np.diag(np.sqrt(covdiag))
        dhatinv = np.diag(1/np.sqrt(covdiag))
        S = dhatinv @ covxx @ dhatinv
    else: 
        S = covxx      
        
    tmp = np.abs(S) - lam
    tmp = (tmp > 0) * tmp
    Ss = np.sign(S) * tmp
    covss = Ss - np.diag(np.diag(Ss)) + np.diag(np.diag(S))
    
    if standard:
        covss = dhat @ covss @ dhat
        
    return covss

def spcovCV(X, standard = False, lambseq = None, K = 10, ntr = None):
    n = X.shape[0]
    covxx = np.cov(X.T) * (n-1)/n
    if ntr is None:
        ntr = np.floor(n*(1-1/np.log(n)))/n
    if standard and (lambseq is None):
        lamseq = np.linspace(0, 0.95, 20)
    elif lambseq is None:     
        abscov = np.abs(covxx)
        lammax = np.max(abscov - np.diag(np.diag(abscov)))
        lamseq = np.linspace(0, lammax, 20)
 
    cvloss = np.zeros((K, 20))
    for k in range(K):
        # print('Fold-', k)
        X_train, X_test = train_test_split(X, test_size=ntr, shuffle = True)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        ss_train = np.cov(X_train.T) * (n_train-1)/n_train
        ss_test = np.cov(X_test.T) * (n_test-1)/n_test
        for i in range(20):
            outcov = spcov(ss_train, lamseq[i], standard)
            cvloss[k, i] = la.norm(outcov-ss_test) ** 2 
            
    l_mean = np.mean(cvloss, axis = 0)
    lambcv = lamseq[np.argmin(l_mean)]
    covss = spcov(covxx, lambcv, standard)
    return covss
