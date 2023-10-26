#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:54:31 2023

@author: wenkaix
"""

import time

start = time.time()

# e2s_app.cond_prob(X)
X = np.ones([1,200,200])
dat = data.DS_Sampled(X)
e2s_app = model.ApproxE2StarStat(dat, n_gen=200)
sampler = model.GlauberSampler(e2s_app, gen_interval=1)        
X = sampler.gen_samples(1, seed=1342+1314*j+5324*k)
end=time.time()

print(end-start)
