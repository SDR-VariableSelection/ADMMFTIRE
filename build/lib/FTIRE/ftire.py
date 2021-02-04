#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
## Example
import FTIRE
from FTIRE import *
X = genxy.generateX(100, 50, 0)
y, d, q, b = genxy.generateY(X, 1)
B = ftire.CV(X, y, d, 30, "ft")

"""

import numpy as np
import scipy.linalg as la
import sys
import dcor
from sklearn.model_selection import KFold # import KFold
import FTIRE

ft = FTIRE.FT.FT
sir = FTIRE.SIR.SIR

__all__ = ["estimate", "CV"]


def projloss(B, b):
    """
    Projection Distance between B and b matrices
    Args:
        B: p times d matrix
        b: p times d estimated matrix
    Returns:
        projection norm loss
    """
    if np.linalg.cond(B) < 1/sys.float_info.epsilon and np.linalg.cond(b) < 1/sys.float_info.epsilon:
#np.isfinite(np.linalg.cond(B)) and np.isfinite(np.linalg.cond(b)):
        loss = la.norm(B @ la.solve(B.T @ B, B.T) - b @ la.solve(b.T @ b, b.T), 2)
    else:
        loss = 100
    return(loss)

def corrloss(B, b, sigt = None, sigs = None):
    """
    distance correlation between B and b
    Args: 
        B: p times d matrix
        b: p times d estimated matrix
    Returns:
        distance correlation
    """
    if np.linalg.cond(B) < 1/sys.float_info.epsilon and np.linalg.cond(b) < 1/sys.float_info.epsilon: #np.isfinite(np.linalg.cond(B)) and np.isfinite(np.linalg.cond(b)):
        if sigt is None:
            p = b.shape[0]
            sigt = np.diag(np.ones(p))
        if sigs is None:
            p = B.shape[0]
            sigs = np.diag(np.ones(p))
        ## normalize B and b
        B = B @ la.inv(la.sqrtm(B.T @ sigs @ B))
        b = b @ la.inv(la.sqrtm(b.T @ sigt @ b))     
        d = B.shape[1]
        loss = np.trace(la.solve(B.T @ sigs @ B, B.T @ sigs @ b) @ la.solve(b.T @ sigs @ b, b.T @ sigs @ B))/d
    else: 
        loss = 0
    return(loss)

def updateC(B, Ups):
    ## update C
    UpsB = (Ups).T @ B
    W1, D, W2 = la.svd(UpsB, full_matrices=False)
    C = W2 @ W1.T ##!
    return(C)


def estimate(X, y, d, m, lamb, method = "ft", NoB = 5, NoC = 20, NoW=2, spX=False, standard=False):
    """
    Estimate B
    Args:
        X: covariates
        y: outcome
        d: structural dimension
        m: number of transfroms
        lamb: regularization parameter
        method: "ft" or "sir"
        NoB: number of iterate over B within ADMM
        NoC: number of iterate over C
        NoW: number of updating weights
        spX: sparse X or not
        standard: standardize X or not
    Returns:
        B: estimate
        covxx: covariance matrix of X
        err2: differences between objective functions since last step
    """
    
    ## init B, C, Ups
    [n,p] = X.shape
    sdr = sir(X = X, y = y, d = d, n_slice = m) if method=='sir' else ft(X = X, y = y,d = d, m = m)
    Ups = sdr.kernel() # phi: p X 2m, My: n X 2m
    # initial B
    # ft.kernel()
    M = sdr.M
    _, B = la.eigh(M, eigvals = (p-d,p-1))
    ## update C
    C = updateC(B, Ups)
    
    ## Covariance
    covxx = FTIRE.SpCov.spcovCV(X, standard = standard) if spX else np.cov(X.T)
    ## initial weight
    weight = np.ones(p)
    
    def updateB(C, weight, lamb):
        rho = 1
        Z = np.zeros((p,d))
        U = np.zeros((p,d))
        for i in range(NoB):
            ## update B
            B = la.solve(covxx + rho * np.eye(p), Ups @ C.T + rho*Z - rho*U) #speed up 
            ## update Z
            Zold = Z.copy()
            lambj = lamb * weight 
            K = np.maximum(1 - lambj/(rho* la.norm(B + U, axis = 1)), np.zeros(p))
            Z = np.diag(K) @ (B+U)
            ## update U
            U = U + (B-Z) 
            #stop criteria
            epsabs = 1e-4
            epsrel = 1e-4
            epspri = np.sqrt(p) * epsabs + epsrel * max(la.norm(B), la.norm(Z))
            epsdual = np.sqrt(p) * epsabs + epsrel * la.norm(U)            
            # err = max(la.norm(B-Z), la.norm(rho*(Zold-Z)))
            # if i == 99:
            #     print('admm %i times' %100)
            if (la.norm(B-Z) < epspri) and (la.norm(rho*(Zold-Z)) < epsdual):
                # print('admm %i times' %i)
                break  
        return(Z)
    

    err2 = np.zeros((NoW,NoC+1))
    for j in range(NoW):
        oldB = B
        err2[j, 0] = np.trace( - Ups.T @ B @ C + C.T @ B.T @ covxx @ B @ C) + lamb * np.dot(weight[np.logical_not(np.isinf(weight))], la.norm(B[np.logical_not(np.isinf(weight)),:], axis = 1))
        for k in range(NoC): 
            ## update B
            B = updateB(C, weight, lamb)
            ## update C
            C = updateC(B, Ups)
            ## stop criteria 
            err2[j, k+1] = np.trace( - Ups.T @ B @ C + C.T @ B.T @ covxx @ B @ C) + lamb * np.dot(weight[np.logical_not(np.isinf(weight))], la.norm(B[np.logical_not(np.isinf(weight)),:], axis = 1))
            # err2 = np.append(err2, np.trace( - Ups.T @ B @ C + C.T @ B.T @ covxx @ B @ C) + lamb * np.dot(weight[np.logical_not(np.isinf(weight))], la.norm(B[np.logical_not(np.isinf(weight)),:], axis = 1)) )
            
            
            if np.isclose(err2[j, k], 0):
                break
            else:
                if abs(err2[j, k+1] - err2[j, k])/abs(err2[j, k]) < 1e-4:
                    break
        # update weight
        Bnorm = la.norm(B, axis = 1)
        close0 = np.isclose(Bnorm, np.zeros(p))
        weight[close0] = np.inf
        no0 = np.logical_not(close0)
        weight[no0] = (1/Bnorm[no0]) ** 1/2
        weight[no0] = weight[no0]/min(weight[no0]) if sum(no0) > 1 else weight[no0]
        # stop criteria for updating weight
        if np.isclose(la.norm(oldB), 0):
            break
        else: 
            if projloss(oldB,B)/la.norm(oldB) < 1e-4:
                # print('update weight %i times' %j)
                break
    return B, covxx, err2

# cross validation
def CV(X, y, d, m, method="ft", nolamb = 50, nofold=10, NoB = 5, NoC = 20, NoW=2, spX=False, standard=False):  
    """
    Estimate B using the best lambda with cross-validation
    Args:
        X: covariates
        y: outcome
        d: structural dimension
        m: number of transfroms
        method: "ft" or "sir"
        nolamb: the number of lambda
        nofold: the number of fold
        NoB: number of iterate over B within ADMM
        NoC: number of iterate over C
        NoW: number of updating weights
        spX: sparse X or not
        standard: standardize X or not
    Returns:
        B: estimate
        covxx: covariance matrix of X
        lambcv: best lambda
        maximized loss 
    """
    ## par.
    #method = 'sir' # or 'ft'
    #nofold = 10
    #nolamb = 50
    #spX = False 
    #standard = False
    #NoB = 5
    #NoC = 20
    #NoW = 2
    
    ## generate lambda candidate
    lambmax = 1 #np.max(sdr0.M)/10
    lambmin = lambmax/1000 if method == 'sir' else lambmax/10
    lambseq = np.exp(np.linspace(np.log(lambmin), np.log(lambmax), num=nolamb))
    
    kf = KFold(n_splits=nofold)
    cvloss = np.zeros((nofold, nolamb))
    k = 0
    for train_index, test_index in kf.split(X):
        print('Fold-', k)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print("TRAIN:", (X_train.shape), "TEST:", (X_test.shape))
        
        for i in range(nolamb):
            Btrain = estimate(X_train,
                              y_train,
                              d,
                              m,
                              lambseq[i],
                              method,
                              NoB,
                              NoC,
                              NoW,
                              spX,
                              standard
                              )[0] # estimate(X, y, d, m, lamb, method = "ft", NoB = 5, NoC = 20, NoW=2, spX=False, standard=False)
            if np.linalg.cond(Btrain) < 1/sys.float_info.epsilon:
                sigs = np.cov(X_train.T)
                Btrain = Btrain @ la.inv(la.sqrtm(Btrain.T @ sigs @ Btrain))
                cvloss[k, i] = 1 - dcor.distance_correlation(X_test @ Btrain, y_test)/d
            else: 
                cvloss[k, i] = 100
        k = k + 1
    l_mean = np.mean(cvloss, axis = 0)
    lambcv = lambseq[np.argmin(l_mean)]
    B, covxx, err= estimate(X, y, d, m, lambcv, method, NoB, NoC, NoW, spX, standard)
    return B, covxx, lambcv, np.argmax(l_mean)

