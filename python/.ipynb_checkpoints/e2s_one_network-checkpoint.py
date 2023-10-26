#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:52:33 2023

@author: wenkaix
"""


import numpy as np
# import matplotlib.pyplot as plt
# import time

import sys
sys.path.append("../")

# import kernel
import utils
import data
# from data import DS_ERGM
# import tests
import model

# import networkx as nx
# import igraph
# import graphkernels as gk

from scipy.spatial.distance import hamming

import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

# from tqdm import tqdm 
# from functools import partial
# import multiprocessing as mp
# from typing import List, Dict

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--d", type=int, default=20)
parser.add_argument("--n_sim", type=int, default=100)
parser.add_argument("--n_test", type=int, default=60)
# parser.add_argument("--test", type=str, default="Approx")
# parser.add_argument("--n_gen", type=int, default=1000)
parser.add_argument("--k", type=int, default=10)
args = parser.parse_args()

r = ro.r
r.source("../Rcode/sim_ergm.R")
r.source("../Rcode/utils.R")
rpy2.robjects.numpy2ri.activate()

d = args.d
n = args.n_sim
k_gen = args.k #number of steps for SteinGen

b = 16 #batch size sampled
n_cpus = 16

coef2 = np.array([-2.,1./float(d)])
e2s_model = model.E2sModel(d, coef2)
dat_e2s = data.DS_ERGM(d, r.construct_e2s_model, coef2)

dat_ds_e2s = dat_e2s.sample(n, return_adj=False)
Xsamples = dat_ds_e2s.X
# true_mean = Xsamples.mean()

# names = ["SteinGen1", "SteinGen_d", "SteinGen_d2", "MPLE", "MLE"]
names = ["SteinGen1",  "MPLE", "MLE"]


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

init_den, init_s2 = compute_init_stat(Xsamples)

ham_dist={}; den_stat={}; s2_stat={}; param_est = {}

for i, name in enumerate(names):
    ham_dist[name] = np.zeros([n,b])
    den_stat[name] = np.zeros([n,b])
    s2_stat[name] = np.zeros([n,b])
    if name[0] == "M": 
        param_est[name] = np.zeros([n,2])


def generate_samples(inputX, method_name, b, return_gen=False):
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
            s2[j] = (X2.sum() - np.diagonal(X2).sum())#/float(len(Xgen[j,:,:]))
            
    else:
        Xgen = np.zeros([b, d, d])
        if method_name == "SteinGen1":
            gen_size = 1
        elif method_name == "SteinGen_d":
            gen_size = d
        elif method_name == "SteinGen_d2":
            gen_size = d*d
        
        for j in range(b):
            X = np.copy(inputX)            
            for k in range(k_gen):
                dat = data.DS_Sampled(X)
                e2s_app = model.ApproxE2StarStat(dat, n_gen=20)
                sampler = model.GlauberSampler(e2s_app, gen_interval=gen_size)        
                X = sampler.gen_samples(1, seed=1342+1314*j+5324*k)
            Xgen[j,:,:] = X[0,:,:]

            ham[j] = hamming(Xgen[j,:,:].reshape([-1]), inputX.reshape([-1]))
            den[j]=(Xgen[j,:,:].mean())
            X2 = np.einsum("ijk, ilk -> ijl", Xgen[j:j+1,:,:], Xgen[j:j+1,:,:]) 
            s2[j] = (X2.sum() - np.diagonal(X2).sum())#/float(len(Xgen[j,:,:]))
            
    if return_gen:
        return ham, den, s2, MC_coef, Xgen
    else:
        return ham, den, s2, MC_coef



# seed = 13423
# with utils.NumpySeedContext(seed=seed):
#     for j, name in enumerate(names):
#         fn = partial(generate_samples, method_name=name, b=b)
#         with mp.Pool(n_cpus) as pool:
#             Xgen = pool.map(fn, [Xsamples[i,:,:] for i in range(n)])
            
for j, name in enumerate(names):
    for i in range(n):
        seed = 13423 + 352*i
        with utils.NumpySeedContext(seed=seed):
            ham, den, s2, MC_coef = generate_samples(Xsamples[i:i+1,:,:], name, b)
            ham_dist[name][i,:] = ham
            den_stat[name][i,:] = den
            s2_stat[name][i,:] = s2
            if name[0] == "M": 
                param_est[name][i,:] = MC_coef
    print(name,"finish")
    np.savez("../res/e2s_one_network_k"+str(k_gen)+"n"+str(d)+".npz", ham_dist = ham_dist, den_stat=den_stat, 
             s2_stat=s2_stat, names=names, init_den = init_den, init_s2 = init_s2, param_est = param_est)
