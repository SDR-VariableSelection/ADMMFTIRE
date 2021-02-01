#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:21:46 2021

@author: jiayingweng
"""

import numpy as np
import scipy.linalg as la

__all__ = ['generateX', 'generateY']

def generateX(n, p, covstr):  
    """
    Generate X for simulation
    Args:
        n (int): sample size
        p (int): number of dimension of X
        covstr (0-3): covariance structure
    Returns:
        X: n times p array
    """
    ## generate X 
    if covstr == 0:
        covx = np.eye(p)
    elif covstr == 1:
        v = 0.5 ** np.arange(p)
        covx = la.toeplitz(v)
    elif covstr == 2:
        offdiag = 0.2
        covx = np.ones((p,p)) * offdiag
        covx = covx + np.eye(p) * (1-offdiag)  
    elif covstr == 3:
        v = 0.8 ** np.arange(p)
        covx = la.toeplitz(v)

    L = np.linalg.cholesky(covx)
    Z = np.random.randn(p,n)
    X = (L @ Z).T  
    return(X)

def generateY(X, M):
    """
    Generate Y based on X
    Args: 
        X: input covariate
        M: model 1-7 uni; 10-15 multi
    Returns:
        Y: outcome
        d: structural dimension
        p: the dimension of Y
        b: the true beta
    """
    [n,p] = X.shape
    ## generate Y      
    if M == 1: # Qian M1
        d = 1
        q = 1
        b = np.zeros((p,d))
        y = np.zeros((n,q))
        index = np.arange(5)
        b[index,:] = 1
        y[:,0] = np.exp(X @ b[:,0]) + np.random.randn(n)
    elif M == 2: # Qian M2 
        d = 2
        q = 1
        b = np.zeros((p,d))
        y = np.zeros((n,q))
        index1 = np.arange(4) #np.random.randint(p, size = 5)
        index2 = np.arange(p-4,p)
        b[index1,0] = 1
        b[index2, 1] = 1
        y[:,0] = np.sign(X @ b[:,0]) * np.log( np.abs( X @ b[:,1] + 5 ) ) + 0.2 * np.random.randn(n)
    elif M == 3: # Tan AOS Model 1
        d = 1
        q = 1
        b = np.zeros((p,d))
        y = np.zeros((n,q))
        index = np.arange(5)
        b[index,:] = 1
        y[:,0] = np.sin(X @ b[:,0]) ** 2 + X @ b[:,0] + np.random.randn(n)
    elif M == 4: # Tan AOS Model 2
        d = 1
        q = 1
        b = np.zeros((p,d))
        y = np.zeros((n,q))
        index = np.arange(5)
        b[index,:] = 1
        y[:,0] = 2 * np.tanh(X @ b[:,0]) + np.random.randn(n)
    elif M == 5: # Cook Zhang 
        d = 1
        q = 1
        b = np.zeros((p,d))
        index = np.arange(1)
        b[index,:] = 1
        X = 1/4 * np.sqrt(0.1) * ( np.random.randn(p,n) + 1) + 1/2 * np.sqrt(0.1) * ( np.random.randn(p,n) + 2 ) + 1/4 * np.sqrt(10) * (np.random.randn(p,n) + 1) 
        X = X.T
        y = np.abs( np.sin( X @ b[:,0] ) ) + 0.2 * np.random.randn(n)
    elif M == 6:
        d = 2
        q = 1
        b = np.zeros((p,d))
        b[0,0] = 1
        b[1,1] = 1
        X[:,1] = X[:,0] + X[:,1]
        X[:,3] = ( 1+X[:,1] ) * X[:,3]
        y = X @ b[:,0] + 0.5 * (X @ b[:,1])** 2   
    elif M == 7:
        d = 2
        q = 1
        b = np.zeros((p,d))
        y = np.zeros((n,q))
        index1 = np.arange(1)
        index2 = np.arange(1,3)
        b[index1,0] = 1
        b[index2, 1] = 1
        y = (X @ b[:,0]) * (X @ b[:,1] + 1) + np.random.randn(n)
    elif M == 10:
        ## simple
        d = 2
        q = 3
        b = np.zeros((p,d))
        y = np.zeros((n,q))
        #index = np.random.randint(p, size = 5)
        index = np.arange(5)
        b[index[0:2], 0] = 1
        b[index[2:], 1] = 1
        y[:,0] = np.exp( X @ b[:,0]) + 0.5 * np.random.randn(n)
        y[:,1] = X @ b[:,1] + 0.1 * np.random.randn(n)
        y[:,2] = 0.1 * np.random.randn(n)
    elif M == 11: ## Zhu Zhu wen 2010 Example 3
        ## complex
        d = 2
        q = 5
        
        covy = np.diag([1,1/2,1/2,1/3,1/4])
        covy[0,1] = covy[1,0] = -1/2
        L = np.linalg.cholesky(covy)
        Z = np.random.randn(q,n)
        eps = (L @ Z).T
    
        b = np.zeros((p,d))
        y = np.zeros((n,q))
        index = np.arange(3) #np.random.randint(p, size = 5)
        b[index[0], 0] = 1
        b[index[1:], 1] = 1
        y[:,0] = 1 + X @ b[:,0] + np.sin(X @ b[:,1]) +  eps[:,0]
        y[:,1] = X @ b[:,1] / (0.5 + (X @ b[:,0])**2) + eps[:,1]
        y[:,2] = np.abs(X @ b[:,1]) * eps[:,2]
        y[:,3] = eps[:,3]
        y[:,4] = eps[:,4]
    elif M == 12: ## Zhu Zhu wen 2010 Example 2 and Bing Li 2008 Model 4.3
        d = 1
        q = 2
        b = np.zeros((p,d))
        b[0:2,0] = [0.8, 0.6]
        top = np.ones((n,2))
        top[:,1] = np.sin(X @ b[:,0])
        y = np.zeros((n,q))
        for i in range(n):
            covy = la.toeplitz(top[i,:])
            L = np.linalg.cholesky(covy)
            Z = np.random.randn(q,1)
            y[i,:] = (L @ Z).T
    elif M == 13: # Bing Li, Weng, Li 2008 Model 4.1
        d = 2
        q = 4
        covy = np.diag([1,1,1,1])
        covy[0,1] = covy[1,0] = -1/2
        L = np.linalg.cholesky(covy)
        Z = np.random.randn(q,n)
        eps = (L @ Z).T
        b = np.zeros((p,d))
        y = np.zeros((n,q))
        index = range(3)
        b[index[0:1], 0] = 1
        b[index[1:], 1] = [2,1]
        
        y[:,0] = X @ b[:,0] + eps[:,0]
        y[:,1] = X @ b[:,1] + eps[:,1]
        y[:,2] = eps[:,2]
        y[:,3] = eps[:,3]
    elif M == 14: # Bing li 2008 Model 4.2
        d = 1
        q = 4
        b = np.zeros((p,d))
        b[0:2,0] = [0.8, 0.6]
        top = np.sin(X @ b[:,0])
        y = np.zeros((n,q))
        for i in range(n):
            covy = np.eye(q)
            covy[0,1] = covy[1,0] = top[i]
            L = np.linalg.cholesky(covy)
            Z = np.random.randn(q,1)
            eps = (L @ Z).T
            y[i,:] = eps
            y[i,0] = np.exp(eps[0,0])
    elif M == 15: # Bing Li 08 Model 4.4
        d = 2
        q = 5
        
        covy = np.diag([1,1/2,1/2,1/3,1/4])
        covy[0,1] = covy[1,0] = -1/2
        L = np.linalg.cholesky(covy)
        Z = np.random.randn(q,n)
        eps = (L @ Z).T
    
        b = np.zeros((p,d))
        y = np.zeros((n,q))
        index = np.arange(5) #np.random.randint(p, size = 5)
        b[index[0:2], 0] = 1
        b[index[2:], 1] = 1
        y[:,0] = X @ b[:,0] + X @ b[:,1] / (0.5 + (X @ b[:,0])**2) + eps[:,0]
        y[:,1] = X @ b[:,0] + np.exp( 0.5 * X @ b[:,1]) +  eps[:,1]
        y[:,2] = X @ b[:,0] + X @ b[:,1] + eps[:,2]
        y[:,3] = eps[:,3]
        y[:,4] = eps[:,4]
    return y, d, q, b
