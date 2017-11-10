# coding: utf-8

import h5py
import logging
import time
import numpy as np
from numpy import dot, zeros, array, eye, kron, prod
from numpy.linalg import norm, solve, inv, svd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from numpy.random import rand

__version__ = "0.5"
__all__ = ['als']

_DEF_MAXITER = 100
_DEF_INIT = 'random'
_DEF_CONV = 1e-4
_DEF_LMBD1 = 0
_DEF_LMBD2 = 0
_DEF_LMBD3 = 0
_DEF_RHO = 0
_DEF_NO_FIT = 1e9
_DEF_FIT_METHOD = None

_log = logging.getLogger('ADMM')

def als(X, rank, **kwargs):

    # ------------ init options ----------------------------------------------
    ainit = kwargs.pop('init', _DEF_INIT)
    maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    lmbda1 = kwargs.pop('lambda_1', _DEF_LMBD1)
    lmbda2 = kwargs.pop('lambda_2', _DEF_LMBD2)
    lmbda3 = kwargs.pop('lambda_3', _DEF_LMBD3)
    rho = kwargs.pop('rho',_DEF_RHO)
    func_compute_fval = kwargs.pop('compute_fval', _DEF_FIT_METHOD)
    dtype = kwargs.pop('dtype', np.float)

    # ------------- check input ----------------------------------------------
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    # check frontal slices have same size and are matrices
    sz = X[0].shape
    for i in range(len(X)):
        if X[i].ndim != 2:
            raise ValueError('Frontal slices of X must be matrices')
        if X[i].shape != sz:
            raise ValueError('Frontal slices of X must be all of same shape')
        #if not issparse(X[i]):
            #raise ValueError('X[%d] is not a sparse matrix' % i)

    if func_compute_fval is None:
        if prod(X[0].shape) * len(X) > _DEF_NO_FIT:
            _log.warn('For large tensors automatic computation of fit is disabled by default\nTo compute the fit, call admm.als with "compute_fit=True"\nPlease note that this might cause memory and runtime problems')
            func_compute_fval = None
        else:
            func_compute_fval = _compute_fval

    n = sz[0]
    m = sz[1]
    k = len(X)

    _log.debug(
        '[Config] rank: %d | maxIter: %d | conv: %7.1e | lmbda: %7.1e' %
        (rank, maxIter, conv, lmbdaA)
    )
    _log.debug('[Config] dtype: %s / %s' % (dtype, X[0].dtype))

    # ------- convert X to CSR ------------------------------------------
    for i in range(k):
        if issparse(X[i]):
            X[i] = X[i].tocsr()
            X[i].sort_indices()
    
    # ---------- initialize theta -----------------------------------
    theta_A1 = zeros((n,rank),dtype = dtype)
    theta_A2 = zeros((n,rank),dtype = dtype)
    theta_B1 = zeros((m,rank),dtype = dtype)
    theta_B2 = zeros((m,rank),dtype = dtype)

    # ---------- initialize A1, A2, B1, B2 -----------------------------------
    _log.debug('Initializing A1, A2, B1, B2, A_bar, B_bar')
    if ainit == 'random':
        A1 = array(rand(n, rank), dtype=dtype)
        A2 = array(rand(n, rank), dtype=dtype)
    else:
        raise ValueError('Unknown init option ("%s")' % ainit)

    A_bar = A1
    B1 = _updateB1(X, A, R, P, Z, lmbdaA, orthogonalize)
    B_bar = B1
    B2 = _updateB2(X, A, R, P, Z, lmbdaA, orthogonalize)

    # ------- initialize R ---------------------------------------------
    R = _updateR(X, A1, B1, lmbdaR)

    # precompute norms of X
    normX = [sum(M.data ** 2) for M in X]

    #  ------ compute factorization ------------------------------------------
    fit = fitchange = fitold = f = 0
    exectimes = []
    for itr in range(maxIter):
        tic = time.time()
        fitold = fit
        A1 = _updateA1(X, B1, R, theta_A1, A_bar, lmbda1)
        A2 = _updateA2(X, A, R, P, Z, lmbdaA, orthogonalize)
        B1 = _updateB1(X, A, R, P, Z, lmbdaA, orthogonalize)
        B2 = _updateB2(X, A, R, P, Z, lmbdaA, orthogonalize)
        R = _updateR(X, A, lmbdaR)

        theta_A1 = _updateTheta(theta_A1, A1, A_bar, rho)
        theta_A2 = _updateTheta(theta_A2, A2, A_bar, rho)
        theta_B1 = _updateTheta(theta_B1, B1, A_bar, rho)
        theta_B2 = _updateTheta(theta_B2, B2, B_bar, rho)

        A_bar = _updateBar(A1, A2)
        B_bar = _updateBar(B1, B2)

        # compute fit value
        if func_compute_fval is not None:
            fit = func_compute_fval(X, A, R, lmbdaA, lmbdaR, lmbdaV, normX)
        else:
            fit = np.Inf

        fitchange = abs(fitold - fit)

        toc = time.time()
        exectimes.append(toc - tic)

        _log.debug('[%3d] fval: %0.5f | delta: %7.1e | secs: %.5f' % (
            itr, fit, fitchange, exectimes[-1]
        ))
        if itr > 0 and fitchange < conv:
            break
    return A1, B1, R, f, itr + 1, array(exectimes)

def _updateA1(X, B1, R, theta_A1, A_bar, lmbda1):

    m, rank = A_bar.shape
    F = zeros((n, rank), dtype=A_bar.dtype)
    E = zeros((rank, rank), dtype=A_bar.dtype)

    BtB = dot(B1.T, B1)
    
    for i in range(len(X)):
        F += X[i].dot(dot(B1, R[i].T))
        E += R[i].dot(dot(BtB, R[i].T))
    
    I = (rho + lmbda1) * eye(rank, dtype=A_bar.dtype)
    J = rho * A_bar - theta_A1

    A = solve(I + E.T, (F + J).T).T
    pass

def _updateA2(self, parameter_list):
    pass

def _updateB1(self, parameter_list):
    pass

def _updateB2(self, parameter_list):
    pass

def _updateR(self, parameter_list):
    pass

def _updateTheta(self, parameter_list):
    pass

def _updateBar():
    pass

