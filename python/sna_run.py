#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

import matplotlib.pyplot as plt

import kernel, data, utils, tests, model
from data import DS_ERGM
from utils import SteinGen_run,SteinGen_run_nr, gkss_, compute_gkss, compute_hamming, gen_cell, compute_e2st
import graph_statistics as gs

import scipy.sparse as sp

import multiprocessing as mp
from typing import List, Dict
from functools import partial
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
# parser.add_argument("--k_gen", type=int, default=10)
parser.add_argument("--b", type=int, default=15)
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
# k_gen = args.k_gen

if model_name == "ER":
    coef=np.array([-2.])
    ergm_model = model.ErdosRenyi(d, coef)
    con_model = r.construct_er_model
    est_model = r.estimate_er
    app_model = model.ApproxEdgeStat 
    
elif model_name == "E2S":
    coef=np.array([-2.,1./float(d)])
    ergm_model = model.E2sModel(d, coef)
    con_model = r.construct_e2s_model
    est_model = r.estimate_e2s
    app_model = model.ApproxE2StarStat
elif model_name == "ET":
    coef=np.array([-2.,1./float(d)])
    ergm_model = model.ETModel(d, coef)
    con_model = r.construct_et_model
    est_model = r.estimate_et
    app_model = model.ApproxETriangleStat

elif model_name == "E2ST":
    con_model = r.construct_e2st_model
    est_model = r.estimate_e2st
    coef=np.array([-2.,1./float(d),-1./float(d)])
    ergm_model = model.E2stModel(d, coef)
    app_model = model.ApproxE2STStat
    

interval = int(d**2/20.)
k_length = np.arange(1, 21) * interval 



res_ = np.load("../res/"+str(model_name)+"n"+str(d)+"gen_samples.npz", allow_pickle=True)
res = res_["res"]
Xs = res_["Xs"]
Xs.shape
n, d, _ = Xs.shape




app_method_ = ["MPLE", "CD", "MLE"]
Xsna = np.zeros([3, n, b, d, d])
MC_coef = np.zeros([3, n, len(coef)])
for m, app_method in enumerate(app_method_):
    for i in range(n):
        try:
            MCcoef = est_model(Xs[i,:,:], app_method)    
            MC_coef[m, i,:] = MCcoef
        except:
            MCcoef = MC_coef[m,i-1,:]
            pass
        with utils.NumpySeedContext(seed=134253+23412*i+1241*m):
            Xsna[m,i,:,:,:] = np.array(r.gen_ergm(d, N=b, construct = con_model, coef = MCcoef))

    np.savez("../res/"+str(model_name)+"n"+str(d)+"gen_sna_samples.npz", Xsna = Xsna, Xs = Xs, MC_coef=MC_coef) #res is a 3 x n x b x d x d; 20 for k_length
   
