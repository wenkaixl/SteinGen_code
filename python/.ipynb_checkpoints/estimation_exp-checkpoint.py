#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:40:47 2023

@author: wenkaix
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("../")

import kernel
import utils
import data
from data import DS_ERGM
import tests
import model

import networkx as nx
import igraph
import graphkernels as gk

from scipy.spatial.distance import hamming

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--d", type=int, default=20)
parser.add_argument("--n_sim", type=int, default=100)
parser.add_argument("--n_test", type=int, default=60)
# parser.add_argument("--test", type=str, default="Approx")
# parser.add_argument("--n_gen", type=int, default=1000)
args = parser.parse_args()


r = ro.r
r.source("../Rcode/sim_ergm.R")
r.source("../Rcode/utils.R")
rpy2.robjects.numpy2ri.activate()


d = 20
n = 100

coef = np.array([-2.])
dat_er = data.DS_ERGM(d, r.construct_er_model, coef)
er_model = model.ErdosRenyi(d, coef)
Xsamples = dat_er.sample(n).X

n_vals = np.array([20, 30, 50, 100])
n_cpus = 16
num_reps = 30

def est_param(d, app_model):    
    # dat_er = data.DS_ERGM(d, r.construct_er_model, coef)
    # app_er = model.ApproxEdgeStat(dat_er, n)
    app_model.cond_prob()
    return app_er.coef

coef_mean = []; coef_std = []
for n_ in tqdm(n_vals):
    dat_er = data.DS_ERGM(d, r.construct_er_model, coef)
    app_er = model.ApproxEdgeStat(dat_er, 1)
    fn = partial(est_param)
    with mp.Pool(n_cpus) as pool:
        res = pool.map(fn, [n_ for _ in range(num_reps)])
    coef_mean.append(np.mean(res))
    coef_std.append(np.std(res))
    print(coef_mean[-1], coef_std[-1])