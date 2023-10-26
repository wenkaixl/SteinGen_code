#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## graph kernel class and related functions

import numpy as np
##to load r package in graphkernels and networks
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri


## requires input as igraph list 
import graphkernels as gk
import igraph as ig



def create_ig_graph(X):
    return ig.Graph.Adjacency(X.tolist())


def compute_transition_list(X):
    # X is a single graph adjaceny matrix
    
    P=[ig.Graph.Adjacency(X.tolist())]
    d = len(X)
    idx = np.triu_indices(d, k=1)
    # for i in range(d-1):
    #     for j in range(i+1, d):
    #         x = X
    #         x[i,j] = abs(1-X[i,j])
    #         x[j,i] = x[i,j]
    for l in range(len(idx[0])):
        x = X
        i = idx[0][l]
        j = idx[1][l]
        x[i,j] = abs(1-X[i,j])
        x[j,i] = x[i,j]
        G = ig.Graph.Adjacency(x.tolist())
        P.append(G)
    return P


def compute_sampled_list(X, sample_idx):
    # X is a single graph adjaceny matrix
    
    P=[ig.Graph.Adjacency(X.tolist())]
    d = len(X)
    l = len(sample_idx)
    for i in range(l):
        x = X
        m = sample_idx[i][0]
        n = sample_idx[i][1]
        
        x[m, n] = abs(1-X[m, n])
        x[n, m] = x[m, n]
        
        G = ig.Graph.Adjacency(x.tolist())
        P.append(G)
    return P

def diag_normalized_kernel(K):
    V = np.diag(K) + 1e-8
    D = 1./np.sqrt(V)
    # D = D[:,np.newaxis]
    return np.einsum('i,ij,j->ij', D, K, D)




## Weisfeiler_Lehman Graph Kernel
def computeWLkernel(X, level=3, diag=1, normalize=True):
    d = len(X)
    P = compute_transition_list(X)
    Kmat = gk.kernels.CalculateWLKernel(P, level)
    K = Kmat[0,0] + Kmat[1:,1:]
    Kvec = Kmat[1:,:1]
    K -=  Kvec + Kvec.T
    if normalize:
        K = diag_normalized_kernel(K)
    if diag==0:
        np.fill_diagonal(K, 0)
    return K


# GKSS methods


# Using vvRKHS
def GKSS_conditional(X, q_X, kernel=computeWLkernel, diagonal=1):
    d = len(X)
    idx = np.triu_indices(d, k=1)
    St =  q_X[idx][:,np.newaxis]
    S = X[idx][:,np.newaxis]
    St_vec=(abs(S - St))
    Smat = St_vec @ (St_vec.T)
    Ky_mat = (S*2-1)@(S*2-1).T    
    
    K = kernel(X, diag=diagonal)
    Jkernel = Smat * K * Ky_mat 
  
    nsim = len(S)
    W = np.ones(nsim)*(1./nsim)
    Jkernel_out = Jkernel
    if diagonal==0:
        np.fill_diagonal(Jkernel, 0)
    v_kernel = np.var(S)
    stats_val = nsim*(W.T @ Jkernel @ W) #* np.sqrt(v_kernel)

    return stats_val, Jkernel_out, K, Smat



def GKSS_sampled(X, q_X, sample_idx, gkernel=gk.kernels.CalculateWLKernel, diagonal=1, level=3):
    d = len(X)
    idx = (sample_idx[:,0], sample_idx[:,1])
    St =  q_X[idx][:,np.newaxis]
    S = X[idx][:,np.newaxis]
    St_vec=(abs(S - St))
    Smat = St_vec @ (St_vec.T)
    Ky_mat = (S*2-1)@(S*2-1).T    
    
    P = compute_sampled_list(X, sample_idx)
    Kmat = gkernel(P, level)
    K = Kmat[0,0] + Kmat[1:,1:]
    Kvec = Kmat[1:,:1]
    K -=  Kvec + Kvec.T
    K = diag_normalized_kernel(K)
    
    Jkernel = Smat * Ky_mat * K
  
    nsim = len(S)
    W = np.ones(nsim)*(1./nsim)
    Jkernel_out = Jkernel
    if diagonal==0:
        np.fill_diagonal(Jkernel, 0)
    v_kernel = np.var(S)
    stats_val = nsim*(W.T @ Jkernel @ W) #* np.sqrt(v_kernel)

    return stats_val, Jkernel_out, K, Smat

