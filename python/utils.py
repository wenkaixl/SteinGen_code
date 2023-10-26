#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time 

import torch
from scipy.spatial.distance import hamming
import scipy.sparse as sp

from cell.utils import link_prediction_performance
from cell.cell import Cell, EdgeOverlapCriterion, LinkPredictionCriterion

import multiprocessing as mp
from typing import List, Dict
from functools import partial

import data, model, kernel
# import model

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
     return np.log(1./(1./(X+1e-7) - 1.))
    
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

def compute_e2st_stats(Xsamples):
    n, d, _ = (Xsamples.shape)
    init_den = np.zeros(n)
    init_s2 = np.zeros(n)
    init_t = np.zeros(n)
    for i in range(n):
        X_ = np.copy(Xsamples[i])
        # init_den[i] = X_.mean()
        init_den[i] = remove_diagonal(X_).mean()
        init_s2[i] = compute_2star(X_)
        init_t[i] = compute_triangle(X_)
    return init_den, init_s2, init_t



def compute_2star(X):
    if len(X.shape)==2:
        X2 = X@X.T
        s2 = X2.sum() - np.diagonal(X2).sum()
        
    elif len(X.shape)>2:
        X2 = np.einsum("ijk, ilk -> ijl", X, X) 
        s2 = (X2.sum() - np.diagonal(X2,axis1=1, axis2=2).sum())/float(len(X))
    return s2/2.
    
def compute_triangle(X):
    if len(X.shape)==2:
        X3 = X@X.T@X
        s3 = np.diagonal(X3).sum()
        
    elif len(X.shape)>2:
        X2 = np.einsum("ijk, ilk -> ijl", X, X) 
        X3 = np.einsum("ijk, ilk -> ijl", X2, X)
        s3 = np.diagonal(X3, axis1=1, axis2=2).sum()/float(len(X))
    return s3/6.

def neighbourhood_average(vec, w=None):
    if w is None:
        conv_weight = np.array([0.5,1,0.5])
    else:
        conv_weight = np.array(w)

    if len(vec) >2:
        r = np.convolve(vec, conv_weight)
        r /= 3.        
        r[1] = (vec[0] + vec[1])/2.
        r[-2] = (vec[-2] + vec[-1])/2.
        return r[1:-1]
    else:
        return vec

def remove_diagonal(X):
    if len(X.shape) == 2:
        m, _ = X.shape
        idx = (np.arange(1,m+1) + (m+1)*np.arange(m-1)[:,None]).reshape(m,-1)
    else:
        n, m, _ = X.shape
        idx = (np.arange(1,m+1) + (m+1)*np.arange(m-1)[:,None] + m*m*np.arange(n)[:,None,None]).reshape(n,m,-1)
    Xrd = X.ravel()[idx]
    return Xrd

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



def gen_cell(X, b=20):
    d = X.shape[1]
    Xgen = np.zeros([b,d,d])
    # CELL baseline
    X_ = np.copy(X)
    train_graph = sp.csr_matrix(X_)
    cell_model = Cell(A=train_graph, H=9, 
                      callbacks=[EdgeOverlapCriterion(invoke_every=10, edge_overlap_limit=.5)])
    #train
    cell_model.train(steps=200,
                optimizer_fn=torch.optim.Adam,
                optimizer_args={'lr': 0.01,
                                'weight_decay': 1e-3})
    #generate
    for i in range(b):
        generated_graph = cell_model.sample_graph()
        Xcell = generated_graph.toarray()
        # print(utils.compute_e2st_stats(Xcell[None,:,:]))
        Xgen[i,:,:] = Xcell
    return Xgen

def choose_hamming(ham_dist, w=5, alpha=0.1):
    n = len(ham_dist)
    h_mean = ham_dist[w-n:] 
    h_std=[]
    k = 0
    for i in range(n-w):
        hsd = ham_dist[i:i+w].std()
        h_std.append(hsd)    
        if hsd > h_mean[i]*alpha:
            k += 1
    return k+w



def SteinGen_run(X, k_length, app_model, b=20, reest=1):
    d = X.shape[-1]
    l = len(k_length)
    Xgen = np.zeros([l, b,  d, d])

    dat = data.DS_Sampled(X[None,:,:])

    app_gen = app_model(dat, n_gen=1)
    
    #no selection
    X_ = np.copy(X)
    for i in range(b):
        dat = data.DS_Sampled(X_[None,:,:])
        app_gen = app_model(dat, n_gen=1)
        j=0
        Xnew = X_[None]
        for k in range(max(k_length)):
            sampler = model.GlauberSamplerES(app_gen)
            # Xnew = sampler.gen_samples(1, seed=1342+5324*k + 132446*i)
            for k_gen in range(reest):
                Xnew = sampler.gen_samples_from_X(Xnew, 1, seed=1342+5324*k + 1324*i + 3312*k_gen)
            dat = data.DS_Sampled(Xnew)
            app_gen = app_model(dat, n_gen=20)
            # print(Xnew.sum())
            if k == k_length[j] - 1:
                Xgen[j, i,:,:] = Xnew[0]
                j += 1
                # if i == 0:
                #     print(app_gen.prob_list[2:18])
        # print("finish:", str(i))
    return Xgen


def SteinGen_run_nr(X, k_length, app_model, b=20, reest=None):
    d = X.shape[-1]
    l = len(k_length)
    Xgen = np.zeros([l, b,  d, d])

    dat = data.DS_Sampled(X[None,:,:])

    app_gen = app_model(dat, n_gen=1)
    
    #no selection
    X_ = np.copy(X)
    for i in range(b):
        dat = data.DS_Sampled(X_[None,:,:])
        app_gen = app_model(dat, n_gen=1)
        sampler = model.GlauberSamplerES(app_gen)
        j=0
        Xnew = X_[None]
        for k in range(max(k_length)):
            Xnew = sampler.gen_samples_from_X(Xnew, 1, seed=1342+5324*k + 1324*i)
            # for k_gen in range(reest):
            #     Xnew = sampler.gen_samples_from_X(Xnew, 1, seed=1342+5324*k + 1324*i + 3312*k_gen)
            # print(Xnew.sum())
            if k == k_length[j] - 1:
                Xgen[j, i,:,:] = Xnew[0]
                j += 1
                # if i == 0:
                #     print(app_gen.prob_list[6:10])
        # print("finish:", str(i))
    return Xgen


def SteinGen_run_multi(X, k_length, app_model, b=20, reest=1):
    d = X.shape[-1]
    n = X.shape[0]
    l = len(k_length)
    Xgen = np.zeros([l, b, d, d])

    dat = data.DS_Sampled(X)

    app_gen = app_model(dat, n_gen=1)
    #no selection
    Xnew = np.copy(X)
    ci = int(np.floor(b/n))
    for i in range(ci):
        dat = data.DS_Sampled(Xnew)
        app_gen = app_model(dat, n_gen=1)
        j=0
        for k in range(max(k_length)):
            sampler = model.GlauberSamplerES(app_gen)
            # Xnew = sampler.gen_samples(1, seed=1342+5324*k + 132446*i)            
            for k_gen in range(reest):
                for s in range(n):
                    Xnew[s] = sampler.gen_samples_from_X(Xnew[s:s+1], 1, seed=1342+5324*k + 1324*i + 3312*k_gen + 2653*s)
            dat = data.DS_Sampled(Xnew)
            app_gen = app_model(dat, n_gen=20)
            # print(Xnew.sum())
            if k == k_length[j] - 1:
                Xgen[j, i*n:(i+1)*n,:,:] = Xnew
                j += 1
                # if i == 0:
                #     print(app_gen.prob_list[2:18])
        # print("finish:", str(i))
    return Xgen

def compute_hamming(X, Xgen):
    hamming_dist = abs(Xgen - X).mean(-1).mean(-1)
    return hamming_dist



def compute_hamming_msd(Xgen_,Xs):
    shp = Xgen_.shape
    n = shp[0]; b=shp[1]
    hamming_mean = np.zeros([n,b]); hamming_std = np.zeros([n,b]); 
    for i in range(n):
        hamming_val = compute_hamming(Xs[i], Xgen_[i])
        hamming_mean[i,:] = hamming_val.mean(1)
        hamming_std[i,:] = hamming_val.std(1)
    
    ham_mean = (hamming_mean).mean(0)
    ham_std = (hamming_std).mean(0)
    
    return ham_mean, ham_std

def compute_hamming_msd_multi(Xgen_,Xs):
    shp = Xgen_.shape
    n = shp[0]; b=shp[1]
    hamming_mean = np.zeros([n,b]); hamming_std = np.zeros([n,b]); 
    for i in range(n):
        hamming_val = compute_hamming(Xs[i], Xgen_[i]).mean(-1)
        hamming_mean[i,:] = hamming_val.mean(1)
        hamming_std[i,:] = hamming_val.std(1)
    
    ham_mean = (hamming_mean).mean(0)
    ham_std = (hamming_std).mean(0)
    
    return ham_mean, ham_std



def gkss_(X, ergm_model, sample_size=None):
    X_ = np.copy(X)
    q_X = ergm_model.cond_prob(X_)
    if len(q_X.shape) == 3:
        q_X = q_X[0]
    if sample_size == None:
        stats_val, _, _, _ = kernel.GKSS_conditional(X_, q_X)
    else:
        d= X_.shape[1]
        with NumpySeedContext(seed=13846):
            sample_idx = np.random.randint(0,d,[sample_size, 2])
        # print(sample_idx[:4])
        stats_val, _, _, _ = kernel.GKSS_sampled(X_, q_X, sample_idx)
    return stats_val

def compute_gkss(Xgen, ergm_model, sample_size=None):
    if len(Xgen.shape) == 3:
        Xgen = Xgen[None,:,:,:]
        
    # if len(Xgen.shape) == 4:
    n,m,d,_ = Xgen.shape
    gkss_val = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            X_ = Xgen[i,j,:,:]
            gkss_val[i,j] = gkss_(X_, ergm_model, sample_size)
        print(i)
    return gkss_val


def compute_gkss_mp(Xgen, ergm_model, n_cpus=64, sample_size=None):
    if len(Xgen.shape) == 3:
        Xgen = Xgen[None,:,:,:]
    
    fn = partial(gkss_, ergm_model=ergm_model, sample_size=sample_size)
    n,m,d,_ = Xgen.shape
    Xgen_ = np.reshape(Xgen, [n*m, d, d])
    with mp.Pool(n_cpus) as pool:
        res = pool.map(fn, [Xgen_[j,:,:] for j in range(n*m)])    
    gkss_val = np.array(res).reshape([n,m])
    print("compute gKSS")
    return gkss_val


def compute_e2st(Xgen):
    if len(Xgen.shape) == 3:
        Xgen = Xgen[None,:,:,:]
        
    # if len(Xgen.shape) == 4:
    n,m,d,_ = Xgen.shape
    den_ = np.zeros([n,m])
    s2_ = np.zeros([n,m])
    t_ = np.zeros([n,m])
    for i in range(n):
        res_c = compute_e2st_stats(Xgen[i])
        den_[i,:] = res_c[0]
        s2_[i,:] = res_c[1]
        t_[i,:] = res_c[2]
    return den_, s2_, t_
