#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
#import scipy.stats as stats
#import torch.autograd as autograd
#import torch
#import torch.distributions as dists
import numpy as np
import utils 


import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

import warnings
import logging



class Data(object):
    """
    Network Data class
    X: Pytorch tensor of adjacency matrices.
    """

    def __init__(self, X):
        """
        X: n x n x m Pytorch tensor for network adjacency matrices
        """
        self.X = X
        # self.X = X.detach()
        # self.X.requires_grad = False
        
    
    def dim(self):
        """Return the dimension of X."""
        dx = self.X.shape[1]
        return dx
    
    def n(self):
        return self.X.shape[0]

    def data(self):
        """Return adjacency matrix X"""
        return (self.X)

    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. 
        Return (Data for tr and te)"""
        # torch tensors
        X = self.X
        n, dx = X.shape
        Itr, Ite = utils.tr_te_indices(n, tr_proportion, seed)
        tr_data = Data(X[Itr].detach())
        te_data = Data(X[Ite].detach())
        return (tr_data, te_data)

    def subsample(self, n, seed=87, return_ind = False):
        """Subsample without replacement. Return a new Data. """
        if n > self.X.shape[0]:
            raise ValueError('n should not be larger than sizes of X')
        ind_x = utils.subsample_ind( self.X.shape[0], n, seed )
        if return_ind:
            return Data(self.X[ind_x, :]), ind_x
        else:
            return Data(self.X[ind_x, :])
        
    def clone(self):
        """
        Return a new Data object with a separate copy of each internal 
        variable, and with the same content.
        """
        nX = self.X.clone()
        return Data(nX)

    def __add__(self, data2):
        """
        Merge the current Data with another one.
        Create a new Data and create a new copy for all internal variables.
        """
        copy = self.clone()
        copy2 = data2.clone()
        nX = np.vstack((copy.X, copy2.X))
        return Data(nX)
# end Data class        


class DataSource(object):
    """
    A source of data allowing resampling. Subclasses may prefix 
    class names with DS. 
    """

    @abstractmethod
    def sample(self, n, seed):
        """Return a Data. Returned result should be deterministic given 
        the input (n, seed)."""
        raise NotImplementedError()

    def dim(self):
       """
       Return the dimension of the data.  If possible, subclasses should
       override this. Determining the dimension by sampling may not be
       efficient, especially if the sampling relies on MCMC.
       """
       dat = self.sample(n=1, seed=3)
       return dat.dim()

#  end DataSource

class DS_ERGM(object):
    """
    A DataSource implementing exponential random graph model.
    """
    def __init__(self, d, construct_model, coef):
        """
        d: size of the network
        construct_model: the ERGM model
        coef: coefficient of network statistics
        """
        self.d = d
        self.coef = coef
        self.construct_model = construct_model
        
    def sample(self, n, seed=3, return_adj = False):
        d = self.d
        r = ro.r
        # import pdb; pdb.set_trace()
        # r.source("../sim_ergm.R")
        # rpy2.robjects.numpy2ri.activate()

        with utils.NumpySeedContext(seed=seed):
            adj_mat = r.gen_ergm(d, n, self.construct_model, self.coef)
        X = np.array(adj_mat)
        if len(X.shape) ==2:
            # This can happen if d=1
            X = X[np.newaxis, :]
        #X = torch.from_numpy(X) #cpu device by default
        if return_adj:
            return Data(X), adj_mat
        else:
            return Data(X)
            # return X

class DS_Trained(object):
    """
    A DataSource implementing a customised pre-trained models.
    """
    def __init__(self, model, d):
        """
        d: size of the network
        model: the pre-trained model that can generate samples 
        """
        self.d = d
        self.model = model
        
    def sample(self, n, seed=3, return_adj = False):
        X = self.model.sample(n)
        adj_mat = np.array(X)
        if return_adj:
            return Data(X), adj_mat
        else:
            return Data(X)
            

class DS_Sampled(object):
    """
    A DataSource implementing a customised pre-trained models in sample forms.
    """
    def __init__(self, model_samples):
        """
        d: size of the network
        model: the pre-trained model that can generate samples 
        """
        self.model_samples = model_samples
        
    def sample(self, n, seed=3, return_adj = False):
        n = min(n, len(self.model_samples))
        X = self.model_samples[:n]
        adj_mat = X
        if return_adj:
            return Data(X), adj_mat
        else:
            return Data(X)

