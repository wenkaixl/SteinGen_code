#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

parser.add_argument("--d", type=int, default=20)
parser.add_argument("--n_sim", type=int, default=20)
# parser.add_argument("--k_gen", type=int, default=10)
parser.add_argument("--b", type=int, default=15)
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
# k_gen = args.k_gen
r_sample= args.r

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
    

interval = int(d**2/40.)
k_length = np.arange(1, 21) * interval 






mean_res_ = []; std_res_ = []
mean_sna_ = []; std_sna_ = []

# r_sample = 1
# res_ = np.load("../res/"+str(model_name)+"n"+str(d)+"r"+str(r_sample)+"gen_multi_samples_overall.npz", allow_pickle=True)
# Xgen_ = res_["res"]
# Xsna = res_["Xsna"]
# Xs = res_["Xs"]



r_list = [1,5,10]
Xgen_list = []
Xs_list = []
for r_sample in r_list:
    print(r_sample)
    res_ = np.load("../res/"+str(model_name)+"n"+str(d)+"r"+str(r_sample)+"gen_multi_samples_overall.npz", allow_pickle=True)
    Xgen_ = res_["res"]
    Xsna = res_["Xsna"]
    Xs = res_["Xs"]
    Xgen_list.append(Xgen_)
    Xs_list.append(Xs)
    
    ci = int(np.floor(b/r_sample))
    mean_res = np.zeros([n,ci]); std_res=np.zeros([n,ci])
    mean_sna = np.zeros([3,n,ci]); std_sna=np.zeros([3,n,ci])
    for i in range(n):
        Xs_ = Xs[i*r_sample:(i+1)*r_sample,:,:]
        Xgen1 = Xgen_[i]
        for c in range(ci):
            hamming_val = utils.compute_hamming(Xgen1[:,c*r_sample:(c+1)*r_sample,:,:],Xs_[None])
            mean_res[i,c] = hamming_val.mean()
            std_res[i,c] = hamming_val.std()
            
            for s in range(3):
                hamming_sna = utils.compute_hamming(Xsna[s,:,c*r_sample:(c+1)*r_sample,:,:],Xs_[None])
                mean_sna[s,i,c] = hamming_sna.mean()
                std_sna[s,i,c] = hamming_sna.std()
            print(r_sample, i,c)
    mean_res_.append(mean_res.mean(-1)); std_res_.append(std_res.mean(-1))    
    mean_sna_.append(mean_sna.mean(-1)); std_sna_.append(std_sna.mean(-1))



l=3
name_list = r_list[:l]
plt.figure()
for i in range(l):
    plt.plot(k_length, mean_res_[i], marker="x", label= "SteinGen"+str(name_list[i]))
    plt.fill_between(k_length, mean_res_[i]-std_res_[i], mean_res_[i]+std_res_[i], alpha=0.35)

    plt.plot(k_length, mean_sna_[i][0], marker="o", label="MPLE"+str(name_list[i]))
    # plt.fill_between(k_length, mean_sna_[i][0]-std_sna_[i][0], mean_sna_[i][0]+std_sna_[i][0], alpha=0.35)
    plt.plot(k_length, mean_sna_[i][1], marker="o", label="CD"+str(name_list[i]))
    # plt.fill_between(k_length, mean_sna_[i][1]-std_sna_[i][1], mean_sna_[i][1]+std_sna_[i][1], alpha=0.35)
    plt.plot(k_length, mean_sna_[i][2], marker="o", label="MLE"+str(name_list[i]))
    # plt.fill_between(k_length, mean_sna_[i][2]-std_sna_[i][2], mean_sna_[i][2]+std_sna_[i][2], alpha=0.35)

plt.legend(ncol=3, bbox_to_anchor=(1., 1.34)) 
# plt.savefig("../fig"+str(model_name)+"n"+str(d)+"gen_multi.pdf", bbox_inches='tight')

# model_name="E2S"



l = len(Xgen_list)
for i in range(l):
    Xs = Xs_list[i]
    Xgen_h = Xgen_list[i]
    gkssX = utils.compute_gkss(Xs, ergm_model, 100)
    gkss_gen = utils.compute_gkss(Xgen_[:,9,:,:,:], ergm_model, 100)
    print(gkssX.mean(), gkss_gen.mean())





app_method_ = ["MPLE", "CD", "MLE"]
markers = ["o", "x", "^"]
for r_sample in r_list:
    print(r_sample)
    res_ = np.load("../res/"+str(model_name)+"n"+str(d)+"r"+str(r_sample)+"gen_multi_samples_overall.npz", allow_pickle=True)
    coef_ = res_["MC_coef"]
    

    plt.figure(figsize=(6,4))
    for i in range(3):
        MCcoef = coef_[i]
        plt.scatter(MCcoef[:,0], MCcoef[:,1], marker=markers[i], s=50,label=app_method_[i])
    plt.plot(coef[0], coef[1], "magenta", marker="*", markersize=18, label="True Model")
    plt.legend()
    plt.xlabel(r"$\beta_1$", size=16)
    plt.ylabel(r"$\beta_2$", size=16, rotation=0)
    plt.savefig("../fig/"+str(model_name)+"n"+str(d)+"r"+str(r_sample)+"est.pdf", bbox_inches='tight')
    


