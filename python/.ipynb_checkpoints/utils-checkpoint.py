#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:03:51 2021

@author: wenkaix
"""
import numpy as np
import time 


from scipy.spatial.distance import hamming

import data 
import model

import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

r = ro.r
r.source("../Rcode/sim_ergm.R")
r.source("../Rcode/utils.R")
rpy2.robjects.numpy2ri.activate()


class NumpySeedContext(object):
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)

# end NumpySeedContext


def sigmoid(X):
     return 1./(1+np.exp(-(X)))
    
def logit(X):
     return np.log(1./(1./(X) - 1.))
    
def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X**2, 1)
    sy = np.sum(Y**2, 1)
    D2 =  sx[:, np.newaxis] - 2.0*X.dot(Y.T) + sy[np.newaxis, :] 
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def dist2_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance **squared** matrix of size
    X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X**2, 1)
    sy = np.sum(Y**2, 1)
    D2 =  sx[:, np.newaxis] - 2.0*np.dot(X, Y.T) + sy[np.newaxis, :] 
    return D2


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.
    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and 
        there are more slightly more 0 than 1. In this case, the m
    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)

def compute_init_stat(Xsamples):
    n = len(Xsamples)
    init_den = np.zeros(n)
    init_s2 = np.zeros(n)
    for i in range(n):
        init_den[i]=(Xsamples[i,:,:].mean())
        X2 = np.einsum("ijk, ilk -> ijl", Xsamples[i:i+1,:,:], Xsamples[i:i+1,:,:]) 
        s2 = (X2.sum() - np.diagonal(X2).sum())#/float(len(Xgen[j,:,:]))
        init_s2[i] = (s2)
    return init_den, init_s2


def compute_2star(X):
    if len(X.shape)==2:
        X2 = X@X.T
        s2 = X2.sum - np.diagonal(X2)
        
    elif len(X.shape)>2:
        X2 = np.einsum("ijk, ilk -> ijl", X, X) 
        s2 = (X2.sum() - np.diagonal(X2,axis1=1, axis2=2).sum())/float(len(X))
    return s2
    
def compute_triangle(X):
    if len(X.shape)==2:
        X3 = X@X.T@X
        s3 = np.diagonal(X3).sum()
        
    elif len(X.shape)>2:
        X2 = np.einsum("ijk, ilk -> ijl", X, X) 
        X3 = np.einsum("ijk, ilk -> ijl", X2, X)
        s3 = np.diagonal(X3, axis1=1, axis2=2).sum()/float(len(X))
    return s3

def neighbourhood_average(vec, w=1):
    r = np.convolve(vec, np.ones(1+2*w))
    r /= 3.
    r[1] = (vec[0] + vec[1])/2.
    r[-2] = (vec[-2] + vec[-1])/2.
    return r[w:-w]


def generate_samples(inputX, method_name, b, k_gen=10, return_gen=False, re_est=True, ES=False):
    """

    Parameters
    ----------
    method_name : string, from the list of names
    inputX : 1 x d x d np.array of adjacency matrix observed
    b : int
        number of network samples to generate
    Returns
        density, two-star statistics, hamming distance from input   
        if return_gen is True also return the b x d x d array of b networks generated 
    """
    d = inputX.shape[1]
    ham = np.zeros(b); den = np.zeros(b); s2 = np.zeros(b)
    MC_coef = 0
    if method_name[0]=="M":
        MC_coef = r.estimate_e2s(inputX[0, :,:], method_name)    
        Xgen = np.array(r.gen_ergm(d, N=b, construct = r.construct_e2s_model, coef = MC_coef))
        for j in range(b):
            ham[j] = hamming(Xgen[j,:,:].reshape([-1]), inputX.reshape([-1]))
            den[j]=(Xgen[j,:,:].mean())
            X2 = np.einsum("ijk, ilk -> ijl", Xgen[j:j+1,:,:], Xgen[j:j+1,:,:]) 
            s2[j] = (X2.sum() - np.diagonal(X2,axis1=1, axis2=2).sum())#/float(len(Xgen[j,:,:]))
            
    else:
        Xgen = np.zeros([b, d, d])
        
        for j in range(b):
            X = np.copy(inputX)            
            dat = data.DS_Sampled(X)
            e2s_app = model.ApproxE2StarStat(dat, n_gen=20)
            if ES is True:
                sampler = model.GlauberSamplerES(e2s_app)
            else:
                sampler = model.GlauberSampler(e2s_app)
 
            for k in range(k_gen):
                if re_est is True:
                    dat = data.DS_Sampled(X)
                    e2s_app = model.ApproxE2StarStat(dat, n_gen=20)
                    if ES is True:
                        sampler = model.GlauberSamplerES(e2s_app)
                    else:
                        sampler = model.GlauberSampler(e2s_app)
                X = sampler.gen_samples(1, seed=1342+1314*j+5324*k)
            Xgen[j,:,:] = X[0,:,:]

            if return_gen is False:
                ham[j] = hamming(Xgen[j,:,:].reshape([-1]), inputX.reshape([-1]))
                den[j]=(Xgen[j,:,:].mean())
                X2 = np.einsum("ijk, ilk -> ijl", Xgen[j:j+1,:,:], Xgen[j:j+1,:,:]) 
                s2[j] = (X2.sum() - np.diagonal(X2,axis1=1, axis2=2).sum())#/float(len(Xgen[j,:,:]))
            
    if return_gen is True:
        return Xgen
    else:
        return ham, den, s2, MC_coef
