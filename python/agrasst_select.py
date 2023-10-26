#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("../")


import kernel, data, utils, tests, model
from data import DS_ERGM
from utils import SteinGen_run,SteinGen_run_nr, gkss_, compute_gkss, compute_hamming, gen_cell, compute_e2st
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

r = ro.r
r.source("../Rcode/sim_ergm.R")
r.source("../Rcode/utils.R")
rpy2.robjects.numpy2ri.activate()


d = 20
n = 30
b = 40
k_gen = 10



model_name ="E2S"


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
    coef=np.array([-2.,1./float(d),1./float(d)])
    ergm_model = model.E2stModel(d, coef)

    app_model = model.ApproxE2STStat

with utils.NumpySeedContext(seed=1323):
    X_ = r.gen_ergm(d,1,con_model,coef)
    X = np.copy(X_)
    Xs  = np.array(r.gen_ergm(d, b, con_model,coef))
Xs.mean()
X.mean()

model_name="ET"

gkss_res = []
gkss_std = []
for d in [20, 30, 50, 100]:
# d = 20
    res_ = np.load("../res/"+str(model_name)+"n"+str(d)+"gen_samples.npz", allow_pickle=True)
    res = res_["res"]
    Xgen_ = res[:,0,:,:,:,:]
    # Xgen_h_ = res[:,1,:,:,:,:]
    # Xcell_ = res[:,2,:,:,:,:]
    
    s=9
    l = len(Xgen_)
    gk_min = []; agr_select = []; gk_mean=[]
    for i in range(l):
        dat = data.DS_Sampled(Xgen_[i,s])
        app_gen = app_model(dat, n_gen=1)
        agrasst_val = compute_gkss(Xgen_[i,s], app_gen, sample_size=100)[0]
    
        gkss_val = compute_gkss(Xgen_[i,s], ergm_model, sample_size=100)[0]
    
    
        min_agg = np.argmin(agrasst_val)
        gkss_val_select = gkss_val[min_agg]
        
        gk_mean.append(gkss_val.mean())
        gk_min.append(np.min(gkss_val))
        agr_select.append(gkss_val_select)
    gkss_res.append(np.array([np.mean(gk_mean),np.mean(agr_select),np.mean(gk_min)]))
    gkss_std.append(np.array([np.std(gk_mean),np.std(agr_select),np.std(gk_min)]))


df=pd.DataFrame(gkss_mean)
print(df.to_latex())

df=pd.DataFrame(gkss_std)
print(df.to_latex())


# dat_h = data.DS_Sampled(Xgen_h[0])
# app_gen_h = model.ApproxE2StarStat(dat_h, n_gen=1)
dat_h = data.DS_Sampled(X[0])
app_gen_h = app_model(dat_h, n_gen=1)

start= time.time()
agrasst_h = compute_gkss_mp(Xgen_h, app_gen_h)
end= time.time()
print(end-start)


plt.figure()
plt.plot(k_length, np.min(agrasst_val,1),"r", marker="^", label="SteinGen")
plt.plot(k_length, np.min(agrasst_h,1),"b", marker="o", label="SteinGen_nr")
plt.plot(k_length, k_length*0.+np.mean(gkss_cell,1),"g", linestyle="--", label="CELL")
plt.plot(k_length, k_length*0.+gkss_X,"k", linestyle="-", label="gkss(X)")
plt.title("Min AgraSSt Value")
plt.legend()


min_agg = np.argmin(agrasst_val, 1)
min_agg_h = np.argmin(agrasst_h, 1)
gkss_val_select = gkss_val[np.arange(len(gkss_val)),min_agg]
gkss_h_select = gkss_h[np.arange(len(gkss_h)),min_agg_h]


plt.figure()
plt.plot(k_length, gkss_val_select,"r", marker="^", label="SteinGen")
plt.plot(k_length, gkss_h_select,"b", marker="o", label="SteinGen_nr")
plt.plot(k_length, k_length*0.+np.mean(gkss_cell,1),"g", linestyle="--", label="CELL")
plt.plot(k_length, k_length*0.+gkss_X,"k", linestyle="-", label="gKSS(X)")
plt.axvline(k_length[s], 0, .8, color="magenta")
# plt.plot(600, 0.1, marker="*", color="magenta")
plt.title("gKSS Values with AgraSSt Select")
plt.legend(ncol=2)

