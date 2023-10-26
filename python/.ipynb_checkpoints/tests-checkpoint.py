#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:03:30 2021

@author: wenkaix
"""


import torch.autograd as autograd
import torch
import torch.distributions as dists
import torch.optim as optim
import typing
from scipy.integrate import quad
from scipy.stats import norm
import numpy as np

import time
import kernel
from kernel import GKSS_conditional, GKSS_sampled

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri


def bootstrapper_rademacher(n):
    """
    Produce a sequence of i.i.d {-1, 1} random variables.
    Suitable for boostrapping on an i.i.d. sample.
    """
    return 2.0*np.random.randint(0, 1+1, n)-1.0

def bootstrapper_multinomial(n):
    """
    Produce a sequence of i.i.d Multinomial(n; 1/n,... 1/n) random variables.
    This is described on page 5 of Liu et al., 2016 (ICML 2016).
    """
    import warnings
    warnings.warn('Somehow bootstrapper_multinomial() does not give the right null distribution.')
    M = np.random.multinomial(n, np.ones(n)/float(n), size=1) 
    return M.reshape(-1) - (1.0/float(n))

def bootstrapper_gaussian(n):
    """
    Produce a sequence of i.i.d standard gaussian random variables.
    """
    return np.random.randn(n)


class GKSStest(object):
    """
    kernelized graph Stein statistics test
    model: probability law or generation process for networks
    k: kernel choice
    alpah: test level
    """
    
    def __init__(self, model, kernel=kernel.computeWLkernel, alpha=0.05, n_simulate=200, 
                     seed=13, ustats=False, resampled=False, B=100):
        self.kernel = kernel
        self.model = model
        self.alpha = alpha
        self.n_simulate = n_simulate
        self.seed = seed
        self.ustats = ustats
        self.resampled = resampled
        #sim the null distribution from MC for testing
        sim_stats = np.zeros(n_simulate)
        Xsample = self.model.gen_model.sample(n_simulate).X
        if resampled:
            d = Xsample.shape[1]
            re_idx = np.random.choice(d, (B,2))
            self.re_idx = re_idx
        for i in range(n_simulate):
            sim_stats[i] = self.compute_stat(Xsample[i])
            if i % 50 == 0:
                print("complete"+str(i)+"sim_stats")
        self.sim_stats = sim_stats
    
    
    def perform_test(self, X, return_simulated_stats=False, return_ustat_gram=False):
        """
        perform the Monte-carlo based tests 
        """
        #start track time
        start = time.time()
        
        if return_ustat_gram:
            test_stat, H= self.compute_stat(X, return_gram_mat=True)
        else:
            test_stat= self.compute_stat(X, return_gram_mat=False)
        
        pvalue = np.mean((self.sim_stats > test_stat)*1.0)
        
        end=time.time()
        
        alpha = self.alpha
        results = {'alpha': alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': self.n_simulate,
                 'time_secs': end-start}        
        if return_simulated_stats:
            results['sim_stats'] = self.sim_stats
        if return_ustat_gram:
            results['H'] = H
            
        return results
        
    def compute_stat(self, X, return_gram_mat = False):
        q_X = self.model.cond_prob(X)
        if len(q_X.shape) ==3:
            q_X = q_X[0]
        #stats, JKernel, K, Smat
        
        if self.resampled:
            res = GKSS_sampled(X, q_X, self.re_idx)
        else:
            res = GKSS_conditional(X, q_X, kernel=self.kernel)
        if return_gram_mat:
            return res[0], res[1]
        else:
            return res[0]
        




class Degtest(object):
    """
    goodness-of-fit test for implicit models based on degree variance
    model: probability law or generation process for networks
    alpah: test level
    """
    
    def __init__(self, model, alpha=0.05, n_simulate=200, seed=13):
        self.model = model
        self.alpha = alpha
        self.n_simulate = n_simulate
        self.seed = seed
        #sim the null distribution from MC for testing
        sim_stats = np.zeros(n_simulate)
        Xsample = self.model.gen_model.sample(n_simulate).X
        for i in range(n_simulate):
            sim_stats[i] = self.compute_stat(Xsample[i])
            if i % 50 == 0:
                print("complete"+str(i)+"sim_stats")
        self.sim_stats = sim_stats
    
    
    def perform_test(self, X, return_simulated_stats=False):
        """
        perform the degree based tests. modified from
        Ouadah, S., Robin, S., and Latouche, P. Degree-based goodness-of-fit tests 
        for heterogeneous random graph models: Independent and exchangeable cases. 
        """
        #start track time
        start = time.time()
        
        test_stat= self.compute_stat(X)
        
        quant = np.mean((self.sim_stats > test_stat)*1.0)
        pvalue = min(quant, 1.-quant) #for two-side tests    
        
        # pvalue = np.mean((self.sim_stats > test_stat)*1.0)
        
        end=time.time()
        
        alpha = self.alpha
        results = {'alpha': alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha/2., 'n_simulate': self.n_simulate,
                 'time_secs': end-start}        
        if return_simulated_stats:
            results['sim_stats'] = self.sim_stats
        
        return results
        
    def compute_stat(self, X):
        Deg = np.sum(X, axis=1)
        # mean = np.sum(Deg, axis=0)
        # n = (X.shape)[1]
        # stat = np.mean((Deg-mean)**2, axis=0)
        stat = np.var(Deg)
        return stat
        
    
class MDdegreeTest(object):
    """
    goodness-of-fit test for implicit models based on Mahalanobis distance between degrees
    model: probability law or generation process for networks
    alpah: test level
    """
    
    def __init__(self, model, alpha=0.05, n_simulate=200, seed=13):
        self.model = model
        self.alpha = alpha
        self.n_simulate = n_simulate
        self.seed = seed
        #sim the null distribution from MC for testing
        sim_stats = np.zeros(n_simulate)
        Xsample = self.model.gen_model.sample(n_simulate).X
        for i in range(n_simulate):         
            sim_stats[i] = self.compute_stat(Xsample[i])
            if i % 50 == 0:
                print("complete"+str(i)+"sim_stats")
        self.sim_stats = sim_stats
    
    
    def perform_test(self, X, return_simulated_stats=False):
        """
        perform the degree based tests. modified from
        Lospinoso, J. and Snijders, T. A. Goodness of fit for stochastic actor-oriented models.
        """
        #start track time
        start = time.time()
        
        test_stat= self.compute_stat(X)
        
        pvalue = np.mean((self.sim_stats > test_stat)*1.0)
        
        end=time.time()
        
        alpha = self.alpha
        results = {'alpha': alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': self.n_simulate,
                 'time_secs': end-start}        
        if return_simulated_stats:
            results['sim_stats'] = self.sim_stats
        
        return results
        
    def compute_stat(self, X):
        edge_count = X.sum()
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        n = X.shape[0]
        d = X.shape[1]
        dc2 = d*(d-1)/2.
        #for undirected graph, average with removed diagonals
        prob = edge_count/(float(n) * 2.*dc2) 
        
        r = ro.r
        r.source("../Rcode/utils.R")
        rpy2.robjects.numpy2ri.activate()
        MD_stat = r("MDdegree")
        stat = MD_stat(prob, X[0,:,:])
        stat = np.array(stat)
        return stat
    
    
class TVdegreeTest(object):
    """
    goodness-of-fit test for implicit models based on Graphical Test Hunter et. al 2008
    test statistics is constructed from Total-variational distnace as modified in Xu & Reinert 2021
    model: probability law or generation process for networks
    alpah: test level
    """
    
    def __init__(self, model, alpha=0.05, n_simulate=200, seed=13):
        self.model = model
        self.alpha = alpha
        self.n_simulate = n_simulate
        self.seed = seed
        #sim the null distribution from MC for testing
        sim_stats = np.zeros(n_simulate)
        Xsample = self.model.gen_model.sample(n_simulate, seed=self.seed).X
        deg_list = self.compute_deg_dist(Xsample)        
        self.deg_list = deg_list
        Xsample = self.model.gen_model.sample(n_simulate, seed=1).X
        for i in range(n_simulate):
            sim_stats[i] = self.compute_stat(Xsample[i])
            if i % 50 == 0:
                print("complete"+str(i)+"sim_stats")
        self.sim_stats = sim_stats 
    
    
    def perform_test(self, X, return_simulated_stats=False):
        """
        perform the test with total variation distance of degrees
        Xu W. and Reinert G. A Stein goodness-of-fit test for exponential random graph models
        """
        #start track time
        start = time.time()
        
        test_stat= self.compute_stat(X)
        
        pvalue = np.mean((self.sim_stats > test_stat)*1.0)
        
        end=time.time()
        
        alpha = self.alpha
        results = {'alpha': alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': self.n_simulate,
                 'time_secs': end-start}        
        if return_simulated_stats:
            results['sim_stats'] = self.sim_stats
        
        return results
        
    def compute_stat(self, X):
        deg = self.compute_deg_dist(X)
        stat = 0.5*np.sum(abs(deg-self.deg_list))
        return stat

    
    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        return deg

    def compute_deg_dist(self, X):
        # return the emprical degree distribution of model sample X
        deg_stat = self.DegStats(X)
        # idx, count = np.unique(deg_stat, return_counts = True)
        # loc = np.searchsorted(idx, deg_stat)
        # deg_count = np.bincount(loc.flatten(), X.flatten())
        # deg_prob = deg_count/count
        d = X.shape[1]
        deg_list = np.zeros(d)
        # for i, x in enumerate(idx):
        #     deg_list[int(x)] = deg_prob[i]
        for i in range(d):
            deg_list[i] = (np.mean(deg_stat == i))
        return deg_list

class ParamTest(object):
    """
    goodness-of-fit test for implicit models based on parameter estimation
    model: probability law or generation process for networks
    alpah: test level
    """
    
    def __init__(self, model, alpha=0.05, n_simulate=200, seed=13):
        self.model = model
        self.alpha = alpha
        self.n_simulate = n_simulate
        self.seed = seed
        #sim the null distribution from MC for testing
        sim_stats = np.zeros(n_simulate)
        Xsample = self.model.gen_model.sample(n_simulate).X
        for i in range(n_simulate):
            sim_stats[i] = (self.compute_stat(Xsample[i]))
            if i % 50 == 0:
                print("complete"+str(i)+"sim_stats")
        self.sim_stats = sim_stats
    
    
    def perform_test(self, X, return_simulated_stats=False):
        """
        perform the parameter estimation based tests on degrees. 
        """
        #start track time
        start = time.time()
        
        test_stat= self.compute_stat(X)
        
        # mean_stat = np.mean(self.sim_stats)
        # std = np.std(self.sim_stats)
        # quant = norm.cdf(test_stat, mean_stat, .2*std)
        # pvalue = min(quant, 1.-quant) #for two-side tests        
        
        quant = np.mean((self.sim_stats > test_stat)*1.0)
        pvalue = min(quant, 1.-quant) #for two-side tests      
        # pvalue = np.mean((self.sim_stats > test_stat)*1.0)

        
        end=time.time()
        
        alpha = self.alpha
        results = {'alpha': alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha/2., 'n_simulate': self.n_simulate,
                 'time_secs': end-start}        
        if return_simulated_stats:
            results['sim_stats'] = self.sim_stats
        
        return results
        
    def compute_stat(self, X):
        r = ro.r
        r.source("../Rcode/utils.R")
        rpy2.robjects.numpy2ri.activate()
        Param_stat = r("Param_degree")
        stat = np.array(Param_stat(X, "CD"))
        return (stat) 
    
# class TSTest:
#     """
#     goodness-of-fit test using simulated network sample from the null Ghoshdastidar et. al 2018
#     model: probability law or generation process for networks
#     alpah: test level
#     """
    
#     def __init__(self, model, alpha=0.05, n_simulate=200, seed=13):
#         self.model = model
#         self.alpha = alpha
#         self.n_simulate = n_simulate
#         self.seed = seed
#         #sim the null distribution from MC for testing
#         sim_stats = np.zeros(n_simulate)
#         Xsample = self.model.gen_model.sample(n_simulate).X
#         deg_list = self.compute_deg_dist(Xsample)        
#         self.deg_list = deg_list
#         Xsample = self.model.gen_model.sample(n_simulate, seed=1).X
#         for i in range(n_simulate):
#             sim_stats[i] = self.compute_stat(Xsample[i])
#             if i % 50 == 0:
#                 print("complete"+str(i)+"sim_stats")
#         self.sim_stats = sim_stats 
    