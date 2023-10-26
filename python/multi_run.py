#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("../")


import kernel, data, utils, tests, model
from data import DS_ERGM
from utils import SteinGen_run_multi, gkss_, compute_gkss, compute_hamming, gen_cell, compute_e2st
import graph_statistics as gs

import scipy.sparse as sp
import networkx as nx
import igraph
import graphkernels as gk

import multiprocessing as mp
from typing import List, Dict
from functools import partial
from tqdm import tqdm

from scipy.spatial.distance import hamming
import pandas as pd 

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri


import torch

from cell.utils import link_prediction_performance
from cell.cell import Cell, EdgeOverlapCriterion, LinkPredictionCriterion


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--d", type=int, default=30)
parser.add_argument("--n_sim", type=int, default=20)
parser.add_argument("--k", type=int, default=1)
parser.add_argument("--b", type=int, default=30)
parser.add_argument("--r", type=int, default=5)
parser.add_argument("--model", type=str, default="E2S")
args = parser.parse_args()


r = ro.r
r.source("../Rcode/sim_ergm.R")
r.source("../Rcode/utils.R")
rpy2.robjects.numpy2ri.activate()

d = args.d
n = args.n_sim
model_name = args.model
b = args.b #batch size sampled
k = args.k
r_sample = args.r

if model_name == "ER":
    coef=np.array([-2.])
    ergm_model = model.ErdosRenyi(d, coef)
    con_model = r.construct_er_model
    est_model = r.estimate_er_multi
    app_model = model.ApproxEdgeStat 
    
elif model_name == "E2S":
    coef=np.array([-2.,1./float(d)])
    ergm_model = model.E2sModel(d, coef)
    con_model = r.construct_e2s_model
    est_model = r.estimate_e2s_multi
    app_model = model.ApproxE2StarStat
elif model_name == "ET":
    coef=np.array([-2.,1./float(d)])
    ergm_model = model.ETModel(d, coef)
    con_model = r.construct_et_model
    est_model = r.estimate_et_multi
    app_model = model.ApproxETriangleStat

elif model_name == "E2ST":
    con_model = r.construct_e2st_model
    est_model = r.estimate_e2st_multi
    coef=np.array([-2.,1./float(d),-1./float(d)])
    ergm_model = model.E2stModel(d, coef)
    app_model = model.ApproxE2STStat
    

interval = int(d**2/20.)
k_length = np.arange(1, 21) * interval 

def gen_samples(X, k_length=k_length, app_model=app_model, b=b):    
    # Xgen_h = SteinGen_run_nr(X, k_length, app_model, b=b, reest=1)
    Xgen = SteinGen_run_multi(X, k_length, app_model, b=b, reest=k)
    # Xcell = gen_cell(X, b=b)
    # l = len(k_length)
    # Xcell = Xcell[None,:,:,:].repeat(l,0)
    # return Xgen, Xgen_h, Xcell
    return Xgen





with utils.NumpySeedContext(seed=1323):
    gsim = r.gen_ergm_list(d, n*r_sample, con_model, coef)
    gsim_r = gsim[0]
    Xs = np.array(gsim[1])
    
    
# r.estimate_e2s_multi(gsim[0][:5])

# Xs  = np.array(r.gen_ergm(d, n, con_model, coef))




app_method_ = ["MPLE", "CD", "MLE"]
Xsna = np.zeros([3, n, b, d, d])
MC_coef = np.zeros([3, n, len(coef)])
for m, app_method in enumerate(app_method_):
    for i in range(n):
        try:
            MCcoef = est_model(gsim_r[i*r_sample:(i+1)*r_sample], app_method)    
            MC_coef[m, i,:] = MCcoef
        except:
            MCcoef = MC_coef[m,i-1,:]
            pass
        with utils.NumpySeedContext(seed=134253+23412*i+1241*m):
            Xsna[m,i,:,:,:] = np.array(r.gen_ergm(d, N=b, construct = con_model, coef = MCcoef))

        np.savez("../res/"+str(model_name)+"n"+str(d)+"r"+str(r_sample)+"gen_multi_sna_samples.npz", Xsna = Xsna, Xs = Xs, MC_coef=MC_coef) #res is a 3 x n x b x d x d; 20 for k_length
   

# Xs.mean()
# X.mean()

fn = partial(gen_samples,k_length=k_length, app_model=app_model, b=b)
n_cpus = 16
with mp.Pool(n_cpus) as pool:
    res = pool.map(fn, [ Xs[i*r_sample:(i+1)*r_sample,:,:]  for i in range(n)])

print(str(model_name), d, r_sample, "finish!")
# np.savez("../res/"+str(model_name)+"n"+str(d)+"gen_samples.npz",res = res, Xs = Xs) #res is a n x 3 x 20 x b x d x d; 20 for k_length

#run collection of k sample interval
np.savez("../res/"+str(model_name)+"n"+str(d)+"r"+str(r_sample)+"gen_samples.npz",res = res, Xs = Xs) #res is a n x 3 x 20 x b x d x d; 20 for k_length


np.savez("../res/"+str(model_name)+"n"+str(d)+"r"+str(r_sample)+"gen_multi_samples_overall.npz", Xsna = Xsna, Xs = Xs, MC_coef=MC_coef, res=res) #res is a 3 x n x b x d x d; 20 for k_length
