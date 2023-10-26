#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


import matplotlib.pyplot as plt

import kernel, data, utils, tests, model
from data import DS_ERGM
from utils import SteinGen_run,SteinGen_run_nr, gkss_, compute_gkss, gen_cell, compute_e2st
import graph_statistics as gs

import scipy.sparse as sp

import multiprocessing as mp
from typing import List, Dict
from functools import partial
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

def compute_gkss_mp(Xgen, ergm_model, n_cpus=64, sample_size=None):
    if len(Xgen.shape) == 3:
        Xgen = Xgen[None,:,:,:]
    
    fn = partial(utils.gkss_, ergm_model=ergm_model, sample_size=sample_size)
    n,m,d,_ = Xgen.shape
    Xgen_ = np.reshape(Xgen, [n*m, d, d])
    with mp.Pool(n_cpus) as pool:
        res = pool.map(fn, [Xgen_[j,:,:] for j in range(n*m)])    
    gkss_val = np.array(res).reshape([n,m])
    print("compute gKSS")
    return gkss_val


def extract_gkss(model_name, d=20, c=100, n=50, return_summary=True):
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
        
    res_ = np.load("../res/"+str(model_name)+"n"+str(d)+"gen_samples.npz", allow_pickle=True)
    res = res_["res"]
    Xs = res_["Xs"]
    Xs.shape
    n, d, _ = Xs.shape
    # b = res.shape[3]

    res_sna = np.load("../res/"+str(model_name)+"n"+str(d)+"gen_sna_samples.npz", allow_pickle=True)
    Xsna = res_sna['Xsna']
    
    Xgen = res[10,0,:,:10,:,:]
    Xgen_h = res[10,1,:,:10,:,:]
    Xcell = res[0,2,:,:10,:,:]

    X_list = [Xgen, Xgen_h, Xcell]
    
    
    gkss_mean = []; gkss_std = []; gkss=[]
    
    with utils.NumpySeedContext(seed=1323):
        Xd  = np.array(r.gen_ergm(d, n, con_model, coef))
    gkss_X = compute_gkss_mp(Xd,ergm_model, sample_size=c)
    gkss.append(gkss_X)
    gkss_mean.append(gkss_X.mean())
    gkss_std.append(gkss_X.std())


    
    for m in range(3):
        val = (compute_gkss_mp(Xsna[m], ergm_model, sample_size=c)) 
        gkss.append(val)
        gkss_mean.append(val.mean(-1).mean(-1))
        gkss_std.append(val.std(-1).mean(-1))
        
        
            
    for m in range(len(X_list)):
        val = (compute_gkss_mp(X_list[m], ergm_model, sample_size=c)) 
        gkss.append(val)
        gkss_mean.append(val.mean(-1).mean(-1))
        gkss_std.append(val.std(-1).mean(-1))
        
    gkss_X = compute_gkss_mp(Xs,ergm_model, sample_size=c)
    gkss.append(gkss_X)
    gkss_mean.append(gkss_X.mean())
    gkss_std.append(gkss_X.std())

    if return_summary is True:
        return np.array(gkss_mean), np.array(gkss_std)
    else:
        return gkss

def extract_pvalue(model_name, d=20, c=100, n=50, alpha=0.05):
    gkss = extract_gkss(model_name, d, c=c, n=n, return_summary=False)
    
    gkss_X = gkss[0]
    q = np.quantile(gkss_X[0], 1-alpha)
    print(q)
    l = len(gkss) - 1
    gkss_rej = np.zeros(l)
    for i in range(l):
        gkss_val = gkss[i+1]
        gkss_rej[i] = (gkss_val.flatten()>q).mean()
    return gkss_rej


model_list = ["E2S","ET","E2ST","ER"]
# g_mean_p = np.zeros([4,7])
g_mean_rej = np.zeros([4,7])

for i, model_name in enumerate(model_list):
    g_pval= extract_pvalue(model_name, d,c=100, n=50, alpha=0.05)
    print(g_pval)
    g_mean_rej[i,:] = (g_pval)


    df = pd.DataFrame(g_mean_rej.T)
    print(df.to_latex())
    
    with open('../res/gkss_rej_rate_n'+str(d)+'.txt', 'w') as tf:
         tf.write(df.to_latex())
    
    np.savez("../res/gkss_rej_rate_n"+str(d)+".npz", gkss_pval = g_mean_rej)






    
