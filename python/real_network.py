#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("../")


import kernel, data, utils, tests, model
from data import DS_ERGM
import graph_statistics as gs

import scipy.sparse as sp
import networkx as nx
import igraph
import graphkernels as gk

from scipy.spatial.distance import hamming
import pandas as pd 

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri


r = ro.r
r.source("../Rcode/sim_ergm.R")
r.source("../Rcode/utils.R")
rpy2.robjects.numpy2ri.activate()



d = 36
# n = 30
b = 30
k_gen = 500

# X = np.array(r.sim_teenager())
# X = np.array(r.sim_florentine())

X = (r.sim_lazega())
X.mean()


Xgen = np.zeros([6,b,d,d])

# ## set up E2ST tools
# MC_param = np.zeros([3,3])
# con_model = r.construct_e2st_model
# est_model = r.estimate_e2st

## set up ER tools
MC_param = np.zeros([3,1])
con_model = r.construct_er_model
est_model = r.estimate_er

method_list = ["MPLE", "CD", "MLE"] 
seed = 13423

with utils.NumpySeedContext(seed=seed):
    for i, app_method in enumerate(method_list):
        MCcoef = est_model(X, app_method)
        MC_param[i,:] = MCcoef
        Xgen[i,:,:,:] = np.array(r.gen_ergm(d, N=b, construct = con_model, coef = MCcoef))


# np.savez("../res/teenager_gen_e2st.npz", X = X, Xgen=Xgen, MC_param=MC_param)
# np.savez("../res/florentine_gen_ER.npz", X = X, Xgen=Xgen, MC_param=MC_param)
np.savez("../res/lazega_gen_ER.npz", X = X, Xgen=Xgen, MC_param=MC_param)


## CELL

import torch

from cell.utils import link_prediction_performance
from cell.cell import Cell, EdgeOverlapCriterion, LinkPredictionCriterion
# from cell.graph_statistics import compute_graph_statistics

train_graph = sp.csr_matrix(X)
cell_model = Cell(A=train_graph, H=7,
             callbacks=[EdgeOverlapCriterion(invoke_every=5, edge_overlap_limit=.85)])
#train
cell_model.train(steps=100,
            optimizer_fn=torch.optim.Adam,
            optimizer_args={'lr': 0.1,
                            'weight_decay': 1e-7})
#generate
for i in range(b):
    generated_graph = cell_model.sample_graph()
    Xcell = generated_graph.toarray()
    # print(utils.compute_e2st_stats(Xcell[None,:,:]))
    Xgen[3,i,:,:] = Xcell

## SteinGen
app_model = model.ApproxEdgeStat
# app_model = model.ApproxE2STStat
dat = data.DS_Sampled(X[None,:,:])

app_gen = app_model(dat, n_gen=1)

#SteinGen
for i in range(b):
    dat = data.DS_Sampled(X[None,:,:])
    app_gen = app_model(dat, n_gen=1)
    for k in range(k_gen):
        sampler = model.GlauberSamplerES(app_gen)
        Xnew = sampler.gen_samples(1, seed=1342+5324*k + 1324*i)
        dat = data.DS_Sampled(Xnew)
        app_gen = app_model(dat, n_gen=20)
    print(i, Xnew.mean())
    Xgen[4,i,:,:] = Xnew[0]

Xgen[4].mean()


#nr
for i in range(b):
    dat = data.DS_Sampled(X[None,:,:])
    app_gen = app_model(dat, n_gen=1)
    sampler = model.GlauberSamplerES(app_gen)
    Xnew = np.copy(X[None,:,:])
    for k in range(k_gen):
        Xnew = sampler.gen_samples_from_X(Xnew, 1, seed=1342+5324*k + 1324*i)
    print(i, Xnew.mean())
    Xgen[5,i,:,:] = Xnew[0]

Xgen[5].mean()

np.savez("../res/lazega_gen_ER.npz", X = X, Xgen=Xgen, MC_param=MC_param)


## Analysis


res = np.load("../res/lazega_gen_ER.npz", allow_pickle=True)
Xgen = res['Xgen'][:6]
X = res["X"]


stats_list = []

for s in range(6):
    # e2st_list.append(utils.compute_e2st_stats(Xgen_[s]))
    stats_ = []
    for i in range(b):
        Ai = (sp.csr_matrix(Xgen[s][i]*1.))
        stats_.append(gs.compute_graph_statistics(Ai))
    stats_list.append(stats_)

Ai = (sp.csr_matrix(X*1.))
stats_list.append([gs.compute_graph_statistics(Ai)])

#hamming distance
ham_dist = utils.compute_hamming(Xgen,X)

name_list = ["density","wedge_count","triangle_count","cpl","LCC","assortativity"]
data_mean = np.zeros((len(stats_list), len(name_list)+2))
data_std = np.zeros((len(stats_list), len(name_list)+2))
for i in range(7):
    stats_ = stats_list[i]
    l = len(stats_)
    for j, name in enumerate(name_list):
        sv = np.zeros(l)
        for ll in range(l):
            sv[ll] = stats_[ll][name]
        data_mean[i,j] = sv.mean()
        if i<7:
            data_std[i,j] = sv.std()
    data_mean[i,6] = ham_dist[i].mean()
    data_std[i,6] = ham_dist[i].std()        


#compute AgraSSt

app_model = model.ApproxEdgeStat
dat = data.DS_Sampled(X[None,:,:])
# dat = data.DS_Sampled(Xgen[4])
app_gen = app_model(dat, n_gen=1)
gk_mean=[]; gk_std=[]
for i in range(6):
    gkss_val = utils.compute_gkss(Xgen[i], app_gen, sample_size=100)
    gk_mean.append(gkss_val.mean())
    gk_std.append(gkss_val.std())

    data_mean[i,-1]=gk_mean        
    data_std[i,-1]=gk_std
    
gkss_val = utils.compute_gkss(X[None], app_gen, sample_size=100)
gk_mean.append(gkss_val.mean())
gk_std.append(gkss_val.std())


