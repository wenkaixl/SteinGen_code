#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy 
import scipy.stats as stats
# from sklearn import kernel_ridge
from sklearn.linear_model import LinearRegression

import torch.autograd as autograd
import torch
import torch.distributions as dists
import networkx

import utils 
import data

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

import warnings
import logging

"""
The ERGM based on Stein characterisation; with the approximated models 
"""

def del_t2s(X):
    if len(X.shape)> 2:
        deg = X.sum(axis=1)[:,:,np.newaxis]
        deg_sum= deg + deg.transpose(0,2,1)
        r = np.arange(X.shape[1])
        deg_sum[:, r, r] = 0
    else:
        deg = X.sum(axis=1)[:,np.newaxis]
        deg_sum = deg + deg.T
        np.fill_diagonal(deg_sum, 0)
    return deg_sum



#The difference function for perturbing edge in triangle term
def del_tri(X):
    if len(X.shape)> 2:
        X2 = np.einsum("ijl,ikl ->ijk", X, X)
        r = np.arange(X.shape[1])
        X2[:, r, r] = 0
    else:
        X2 = X@X.T
        np.fill_diagonal(X2, 0)
    return X2




class ErgmModel(with_metaclass(ABCMeta, object)):
    """
    An abstract class of an Exponential Random Graph Model(ERGM).  This is
    intended to be used to represent a model of the data for goodness-of-fit
    testing.
    """

    @abstractmethod
    def t_fun(self, X):
        """
        The Delta_s t(x) for each edge s 
        """
        raise NotImplementedError()

    def log_normalized_prob(self, X):
        """
        Evaluate the exact normalised probability. The normalizer required. 
        This method is not essential. For sanity check when necessary
        Subclasses do not need to override.
        """
        raise NotImplementedError()

    def get_datasource(self):
        """
        Return a DataSource that allows sampling from this model.
        May return None if no DataSource is implemented.
        Implementation of this method is not enforced in the subclasses.
        """
        return None

    
    def cond_prob(self, X):
        '''
        Conditional probability of q(x^{(s,1)}|x_{-s}),  sigmoid(t_fun)
        '''
        return 1./(1+np.exp(-self.t_fun(X)))


    @abstractmethod
    def dim(self):
        """
        Return the dimension of the input.
        """
        raise NotImplementedError()

# end UnnormalizedDensity



class ErdosRenyi(ErgmModel):
    """
    explicit density model for Erdos-Renyi graph
    """
    def __init__(self, d, coef):
        """
        d: size of the network
        coef: coefficient of network statistics
        """
        self.d = d 
        self.coef = coef
        self.gen_model = self.get_datasource()
        self._name = "ERmodel"

    def t_fun(self, X):
        '''
        The Delta_s t(x) for each edge s is independent of other edges;
        returning the coeficient for edge statistics

        '''
        return self.coef * (X*0 + 1.)


    def get_datasource(self):
        r= ro.r
        r.source("../Rcode/utils.R")
        return data.DS_ERGM(self.d, r.construct_er_model, self.coef)

    def dim(self):
        return len(self.d)


class E2sModel(ErgmModel):
    """
    explicit density model for Edge-2Star ERGM
    """
    
    def __init__(self, d, coef):
        """
        d: size of the network
        coef: coefficient of network statistics
        """
        self.d = d 
        self.coef = coef
        self.gen_model = self.get_datasource()
        self._name = "E2Smodel"
        
    def t_fun(self, X):
        '''
        The Delta_s t(x) for each edge s is the number of degrees associated with 
        both vertices of an edge;
        '''
        return self.coef[0] + self.coef[1]*del_t2s(X)


    def get_datasource(self):
        r = ro.r
        r.source("../Rcode/utils.R")
        return data.DS_ERGM(self.d, r.construct_e2s_model, self.coef)

    def dim(self):
        return len(self.d)
    
    
    
class ETModel(ErgmModel):
    """
    explicit density model for Edge-Triangle ERGM
    """
    
    def __init__(self, d, coef):
        """
        d: size of the network
        coef: coefficient of network statistics
        """
        self.d = d 
        self.coef = coef
        self.gen_model = self.get_datasource()
        self._name = "ETmodel"
        
    def t_fun(self, X):
        '''
        The Delta_s t(x) for each edge s is the number of degrees associated with 
        both vertices of an edge;
        '''
        return self.coef[0] + self.coef[1]*del_tri(X)


    def get_datasource(self):
        r = ro.r
        r.source("../Rcode/utils.R")
        return data.DS_ERGM(self.d, r.construct_et_model, self.coef)


    def dim(self):
        return len(self.d)
    

class E2stModel(ErgmModel):
    """
    explicit density model for Edge-2Star-Triangle ERGM
    """
    
    def __init__(self, d, coef):
        """
        d: size of the network
        coef: coefficient of network statistics
        """
        self.d = d 
        self.coef = coef
        self.gen_model = self.get_datasource()
        self._name = "E2STmodel"
        
    def t_fun(self, X):
        '''
        The Delta_s t(x) for each edge s is the number of degrees associated with 
        both vertices of an edge and the number of common edges;
        '''
        return self.coef[0] + self.coef[1]*del_t2s(X) + self.coef[2]*del_tri(X)


    def get_datasource(self):
        r = ro.r
        r.source("../Rcode/utils.R")
        return data.DS_ERGM(self.d, r.construct_e2st_model, self.coef)

    def dim(self):
        return len(self.d)
    


class GlauberSampler():
    """
    Generate samples via mpleglauber dynamics
    """
    def __init__(self, gen_model, gen_interval=1, init=None):
        """
        gen_model: class to generate network samples from requires function cond_prob 
        """
        self.gen_model = gen_model
        self.gen_interval = gen_interval
        self.init = init
        
    def gen_samples(self, n=1, seed=1314):
        # if self.init is None:
        X_old = self.gen_model.sample(1, seed=seed).X #X is n x d x d 
        X = np.copy(X_old)
        X = self.gen_samples_from_X(X, n)
        return X
    
    def gen_samples_from_X(self, X_old, n=1, seed=1314):
        d = X_old.shape[-1]
        X = np.copy(X_old.repeat(n,0))
        # l = self.gen_interval
        with utils.NumpySeedContext(seed=seed):
            for id1 in range(1):
                cond_prob = self.gen_model.cond_prob(X)
                # U = np.random.uniform(0, 1, n*d*d)
                U = np.random.uniform(0, 1, n)              
                idx = np.random.randint(0,d,[n,2])
                idx_diag = (np.where(idx[:,0] == idx[:,1])[0])
                prob_samp = cond_prob[np.arange(n),idx[:,0],idx[:,1]]
                prob_samp[idx_diag] = 0
                Xind = ((prob_samp - U)>0)*1.
                Xind[idx_diag] = 0
                X[np.arange(n),idx[:,0],idx[:,1]] = Xind
                X[np.arange(n),idx[:,1],idx[:,0]] = Xind                
        return X
    
      

class GlauberSamplerES():
    """
    More efficient sampling procedure for generate samples via glauber dynamics
    """
    def __init__(self, gen_model, init=None):
        """
        gen_model: class to generate network samples from; requires function cond_prob 
        """
        self.gen_model = gen_model
        self.init = init
        
    def gen_samples(self, n=1, seed=1314):
        X_old = self.gen_model.sample(n, seed=seed).X #X is n x d x d 
    #     X = self.gen_samples_from_X(X_old, n)
    #     return X
    # def gen_samples_from_X(self, X_old, n=1, seed=1314):
        d = X_old.shape[1]
        X = np.copy(X_old.repeat(n,0))
        with utils.NumpySeedContext(seed=seed):
            for id1 in range(1):
                cond_prob = self.gen_model.cond_prob(X)
                U = np.random.uniform(0, 1, d*d).reshape([d,d])
                X_ = ((cond_prob - U)>0)*1
                Xind = np.where(X != X_)
                if len(Xind[0])>=1:
                    idx = np.random.choice(len(Xind[0]), n)
                    idx = np.unique(idx)
                    n_ = len(idx)
                    # X[0, Xind[1][idx],Xind[2][idx]] = X_[0, Xind[1][idx],Xind[2][idx]]   
                    # X[0, Xind[1][idx],Xind[2][idx]] = X_[0, Xind[1][idx],Xind[2][idx]]   
                    # np.fill_diagonal(X[0], 0)
                    X[np.arange(n_), Xind[1][idx],Xind[2][idx]] = X_[np.arange(n_), Xind[1][idx],Xind[2][idx]]   
                    X[np.arange(n_), Xind[1][idx],Xind[2][idx]] = X_[np.arange(n_), Xind[1][idx],Xind[2][idx]]   
        return X
    
    def gen_samples_from_X(self, X_old, n=1, seed=1314):
        d = X_old.shape[1]
        X = np.copy(X_old.repeat(n,0))
        with utils.NumpySeedContext(seed=seed):
            for id1 in range(1):
                cond_prob = self.gen_model.cond_prob(X)
                U = np.random.uniform(0, 1, d*d).reshape([d,d])
                X_ = ((cond_prob - U)>0)*1
                Xind = np.where(X != X_)
                if len(Xind[0])>=1:
                    idx = np.random.choice(len(Xind[0]), n)
                    idx = np.unique(idx)
                    n_ = len(idx)
                    # X[0, Xind[1][idx],Xind[2][idx]] = X_[0, Xind[1][idx],Xind[2][idx]]   
                    # X[0, Xind[1][idx],Xind[2][idx]] = X_[0, Xind[1][idx],Xind[2][idx]]   
                    # np.fill_diagonal(X[0], 0)
                    X[np.arange(n_), Xind[1][idx],Xind[2][idx]] = X_[np.arange(n_), Xind[1][idx],Xind[2][idx]]   
                    X[np.arange(n_), Xind[1][idx],Xind[2][idx]] = X_[np.arange(n_), Xind[1][idx],Xind[2][idx]]   
        return X


class ApproxModel(with_metaclass(ABCMeta, object)):
    """
    An abstract class of an ERGM based on approximate conditional probability
    This is intended to be used to represent a model of the data for model validation
    """

    @abstractmethod    
    def cond_prob(self, X):
        '''
        The approximate conditional probability of \hat{q}(x^{(s,1)}|x_{-s})
        '''
        raise NotImplementedError()

    def t_fun(self, X):
        """
        The Delta_s t(x) for each edge s; logit q (x^{(s,1)}|x_{-s})
        """
        return np.log(1./(1./self.cond_prob(X)-1.))

    def log_normalized_prob(self, X):
        """
        Evaluate the exact normalised probability. The normalizer required. 
        This method is not essential. For sanity check when necessary
        Subclasses do not need to override.
        """
        raise NotImplementedError()

    def get_datasource(self):
        """
        Return a DataSource that allows sampling from this model.
        May return None if no DataSource is implemented.
        Implementation of this method is not enforced in the subclasses.
        """
        return None

    # @abstractmethod
    # def dim(self):
    #     """
    #     Return the dimension of the input.
    #     """
    #     raise NotImplementedError()


class ApproxEdgeStat(ApproxModel):
    """
    model with approximated conditional edge probability of ER graoph
    """
    def __init__(self, gen_model, n_gen=100, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        # if smooth:
        #     self.EstEdgeProb(self.Xsample.X)
        # else:
        #     self.CountEdgeProb(self.Xsample.X)
        self.est_prob(self.Xsample.X)
        self._name="Approx_ER"

    def sample(self, n=100, seed=1):
        return self.gen_model.sample(n, seed=seed)        
    
    def est_prob(self, X, fit=False):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        edge_count = X.sum()
        n = X.shape[0]
        d = X.shape[1]
        dc2 = d*(d-1)/2.
        #for undirected graph, average with removed diagonals
        prob = edge_count/(float(n) * 2.*dc2) 
        self.prob = prob
        self.coef = utils.logit(prob)
        if fit:
            coef = utils.logit(prob)
            return prob, coef
        else:
            return prob
    
    def cond_prob(self, X):
        return self.prob * (X*0 + 1.)
    
    def t_fun(self, X):
        '''
        The Delta_s t(x) for each edge s is independent of other edges;
        returning the coeficient for edge statistics
        '''
        return self.coef * (X*0 + 1.)



class ApproxE2S(ApproxModel):
    """
    model with approximated conditional edge probability of ER graoph
    """
    def __init__(self, gen_model, n_gen=100):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        self.DegProb(self.Xsample.X)
        self._name="Approx_E2S"

    def sample(self, n=100, seed=1):
        return self.gen_model.sample(n, seed=seed)
    
    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        deg_stat = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_stat -= 2*X
        return deg_stat
    
    def DegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        n, m, _ = deg_stat.shape
        idx = (np.arange(1,m+1) + (m+1)*np.arange(m-1)[:,None] + m*m*np.arange(n)[:,None,None]).reshape(n,m,-1)
        deg_stat_ = deg_stat.ravel()[idx]
        X_ = X.ravel()[idx]

        idx, count = np.unique(deg_stat_, return_counts = True)
        loc = np.searchsorted(idx, deg_stat_)
        deg_count = np.bincount(loc.flatten(), X_.flatten())
        deg_prob = deg_count/count

        deg_count_nz = deg_count[deg_count>0]
        count_nz = count[deg_count>0]
        idx_nz = idx[deg_count > 0]
        
        deg_count_c = utils.neighbourhood_average(deg_count_nz)
        count_c = utils.neighbourhood_average(count_nz)
        deg_prob_c = deg_count_c/count_c

        deg_prob_c = deg_prob_c.clip(min=1e-5, max=1-1e-5)
        logit_dp = (utils.logit(deg_prob_c))

        lr = LinearRegression()
        lr.fit(idx_nz[:,None], logit_dp[:,None])
        self.lr = lr
        
        d = X.shape[1]
        deg_list = np.arange(2*d)
        pred_sig = lr.predict(deg_list[:,None])
        prob_list = utils.sigmoid(pred_sig)
        
        # count_list = np.zeros(2*d)
        # for i, x in enumerate(idx):
        #     prob_list[int(x)] = deg_prob[i]
        #     count_list[int(x)] = count[i]
        # self.count_list = count_list
        self.prob_list = prob_list[:,0]
        self.deg_list = deg_list 
        return prob_list, deg_list
    
    def cond_prob(self, X, fit=False):
        deg_stat = self.DegStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        prob = deg_stat * 0.
    
        for i, deg in enumerate(deg_list):
            prob[deg_stat == deg] = prob_list[i]
        
        if fit:
            return prob, self.lr.coef_
        else:
            return prob
    
    
    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS
    


class ApproxET(ApproxModel):
    """
    model with approximated conditional edge probability with edge and triangle count statistics
    The number of common neighbour vertices is used to estimation conditional probability
    """
    def __init__(self, gen_model, n_gen=100):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        self.CountProb(self.Xsample.X)
        self._name="Approx_ET"
        
    def sample(self, n=100, seed=1):
        return self.gen_model.sample(n, seed=seed)
    

    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        # maximum and mininum degree between two vertices connecting edge=ij
        # remove the edge connecting edge=ij
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        deg_stat = np.array([deg_min_stat, deg_max_stat])
        return deg_stat


    def NeighbourStats(self, X):
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        X2 = np.einsum("ijk, ilk -> ijl", X, X)
        return X2
        # return deg_min_stat, deg_max_stat
    
    def CountProb(self, X):
        #compute the number of common neighbours to form triangles
        neighbour_stat = self.NeighbourStats(X)
        n, m, _ = neighbour_stat.shape
        idx = (np.arange(1,m+1) + (m+1)*np.arange(m-1)[:,None] + m*m*np.arange(n)[:,None,None]).reshape(n,m,-1)
        neighbour_stat_ = neighbour_stat.ravel()[idx]
        X_ = X.ravel()[idx]
        
        
        idx, count = np.unique(neighbour_stat_, return_counts = True)
        loc = np.searchsorted(idx, neighbour_stat_)
        deg_count = np.bincount(loc.flatten(), X_.flatten())
        deg_prob = deg_count/count

        deg_count_nz = deg_count[deg_count>0]
        count_nz = count[deg_count>0]
        idx_nz = idx[deg_count > 0]
        
        deg_count_c = utils.neighbourhood_average(deg_count_nz)
        count_c = utils.neighbourhood_average(count_nz)
        deg_prob_c = deg_count_c/count_c

        deg_prob_c = deg_prob_c.clip(min=1e-5, max=1-1e-5)
        logit_dp = (utils.logit(deg_prob_c))

        lr = LinearRegression()
        lr.fit(idx_nz[:,None], logit_dp[:,None])
        self.lr = lr
        
        d = X.shape[1]
        deg_list = np.arange(d-1)
        pred_sig = lr.predict(deg_list[:,None])
        prob_list = utils.sigmoid(pred_sig)
        
        self.prob_list = prob_list[:,0]
        self.deg_list = deg_list 
        return prob_list, deg_list


    def cond_prob(self, X, fit=False):
        deg_stat = self.NeighbourStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        if len(X.shape)==2:
            prob =  X[np.newaxis,:] * 0. 
        else:
            prob = X * 0.
        
        for i, deg in enumerate(deg_list):
            prob[deg_stat == deg] = prob_list[i]
        
        if fit:
            return prob, self.lr.coef_
        else:
            return prob
        
    
    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS
    
    

class ApproxE2ST(ApproxModel):
    """
    model with approximated conditional edge probability with edge and triangle count statistics
    The number of common neighbour vertices is used to estimation conditional probability
    """
    def __init__(self, gen_model, n_gen=100):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        self.CountProb(self.Xsample.X)
        self._name="Approx_E2ST"
        
    def sample(self, n=100, seed=1):
        return self.gen_model.sample(n, seed=seed)
    

    def DegStats_minmax(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        # maximum and mininum degree between two vertices connecting edge=ij
        # remove the edge connecting edge=ij
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        deg_stat = np.array([deg_min_stat, deg_max_stat])
        return deg_stat

    
    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        deg_stat = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_stat -= 2*X
        return deg_stat
    
    def NeighbourStats(self, X):
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        X2 = np.einsum("ijk, ilk -> ijl", X, X)
        return X2
        # return deg_min_stat, deg_max_stat
    
    def CountProb(self, X):
        #compute the number of common neighbours to form triangles
        neighbour_stat = self.NeighbourStats(X)
        deg_stat = self.DegStats(X)
        n, m, _ = neighbour_stat.shape
        idx = (np.arange(1,m+1) + (m+1)*np.arange(m-1)[:,None] + m*m*np.arange(n)[:,None,None]).reshape(n,m,-1)
        neighbour_stat_ = neighbour_stat.ravel()[idx]
        deg_stat_ = deg_stat.ravel()[idx]
        X_ = X.ravel()[idx]
        
        
        deg_flat = np.array([deg_stat_.flatten(), neighbour_stat_.flatten()])
        idx, loc, count = np.unique(deg_flat, axis=1, return_counts=True, return_inverse=True)
        deg_count = np.bincount(loc, X_.flatten())
        # deg_prob = deg_count/count

        lr = LinearRegression()
        if np.sum(deg_count) > 0:
            deg_count_nz = deg_count[deg_count>0]
            count_nz = count[deg_count>0]
            idx_nz = idx[0][deg_count > 0], idx[1][deg_count>0]
            idx_nz = np.array(idx_nz).T
            
            deg_count_c = utils.neighbourhood_average(deg_count_nz)
            count_c = utils.neighbourhood_average(count_nz)
            deg_prob_c = deg_count_c/count_c
    
            deg_prob_c = deg_prob_c.clip(min=1e-5, max=1-1e-5)
            logit_dp = (utils.logit(deg_prob_c))
    
            lr.fit(idx_nz, logit_dp[:,None])
            self.lr = lr
        else:
            lr.fit(np.ones([10,2]), np.zeros([10,1]))
            self.lr = lr
            
        d = X.shape[1]
        deg_list1 = np.arange(2*d-2)
        deg_list2 = np.arange(d-1)
        deg_list = np.array(np.meshgrid(deg_list1, deg_list2)).reshape([2,-1]).T
        # pred_sig = lr.predict(deg_list)
        # prob_list = utils.sigmoid(pred_sig)
        
        # self.prob_list = prob_list[:,0]
        # self.deg_list = deg_list 
        # return prob_list, deg_list
        return lr


    def cond_prob(self, X, fit=False):
        neighbour_stat = self.NeighbourStats(X)
        deg_stat = self.DegStats(X)
        
        input_stat = np.concatenate([deg_stat.reshape([1,-1]), neighbour_stat.reshape([1,-1])])
        lr = self.lr
        pred = lr.predict(input_stat.T)
        prob = utils.sigmoid(pred)
        
        # prob_list = self.prob_list
        # deg_list = self.deg_list
        if len(X.shape)==2:
            prob =  X[np.newaxis,:] * 0. 
        else:
            prob = X * 0.
        
        prob = prob.reshape(X.shape)
        # for i, deg in enumerate(deg_list):
        #     prob[deg_stat == deg] = prob_list[i]
        
        if fit:
            return prob, self.lr.coef_
        else:
            return prob
        
    
    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS
    
    








class ApproxE2StarStat(ApproxModel):
    """
    model with approximated conditional edge probability of Edge+2Star graph
    """
    def __init__(self, gen_model, n_gen=100, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        self.CountDegProb(self.Xsample.X)
        self.smooth = smooth
        self._name="Approx_E2StarStat"

    def sample(self, n=100, seed=1):
        return self.gen_model.sample(n, seed=seed)

    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        deg_stat = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        #remove diagonal terms
        # np.apply_along_axis(np.fill_diagonal, 0, deg_stat, val=0)
        # remove the presence of edge ij
        deg_stat -= 2*X
        return deg_stat

    
    def CountDegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        idx, count = np.unique(deg_stat, return_counts = True)
        loc = np.searchsorted(idx, deg_stat)
        deg_count = np.bincount(loc.flatten(), X.flatten())
        deg_prob = deg_count/count
        
        #make a lookup table
        # n = X.shape[0]
        d = X.shape[1]
        #total possible degree 0--2(d-2)
        deg_list = np.arange(2*d)
        prob_list = np.ones(2*d) * 0.
        count_list = np.zeros(2*d)
        for i, x in enumerate(idx):
            prob_list[int(x)] = deg_prob[i]
            count_list[int(x)] = count[i]
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.deg_count = deg_count 
        self.count_list = count_list
        return prob_list, deg_list, count_list
    
    # def EstDegProb(self, X, method="krr", weighted=False):
    #     # prob_list, deg_list, count_list = self.CountDegProb(X)
    #     # prob_smooth, deg_smooth, _ = scipy.interpolate.splrep(deg_list,prob_list)
    #     deg_stat = self.DegStats(X)
    #     idx, count = np.unique(deg_stat, return_counts = True)
    #     loc = np.searchsorted(idx, deg_stat)
    #     deg_count = np.bincount(loc.flatten(), X.flatten())
    #     deg_prob = deg_count/count
        
        
    #     n = X.shape[0]
    #     d = X.shape[1]
    #     #total possible degree 0--2(d-2)
    #     deg_list = np.arange(2*d)
        
    #     if method == "krr":
    #         krr = kernel_ridge.KernelRidge(kernel="rbf", gamma=.3) #gaussian kernel default
    #         if weighted:
    #             krr.fit(idx[:,np.newaxis], deg_prob[:,np.newaxis], np.sqrt(deg_count+1e-6))
    #         else:
    #             krr.fit(idx[:,np.newaxis], deg_prob[:,np.newaxis])
    #         prob_list = krr.predict(deg_list[:,np.newaxis])
    #         prob_list = prob_list.clip(min=0.)
    #         prob_list = prob_list.clip(max=1.)

    #         krr_count = kernel_ridge.KernelRidge() #linear for count default
    #         krr_count.fit(idx[:,np.newaxis], deg_count[:,np.newaxis])
    #         count_list = krr_count.predict(deg_list[:,np.newaxis])
    #         count_list = count_list.clip(min=0)
            
    #     self.prob_list = prob_list
    #     self.deg_list = deg_list
    #     self.count_list = count_list
    #     self.prob_pred = krr
    #     return prob_list, deg_list, count_list

    def cond_prob(self, X, smooth=None):
        if smooth is None:
            smooth = self.smooth 
        deg_stat = self.DegStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        prob = deg_stat * 0.

        for i, deg in enumerate(deg_list):
            prob[deg_stat == deg] = prob_list[i]
        
        return prob


    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS
    
    


class ApproxE2StarStatCumulative(ApproxE2StarStat):
    """
    model with approximated conditional edge probability of Edge+2Star graph
    The cumulative notion is used to estimation conditional probability
    """
    def __init__(self, gen_model, n_gen=100, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        if smooth:
            self.EstDegProb(self.Xsample.X)
        else:
            self.CountDegProb(self.Xsample.X)
        self.smooth = smooth
        self._name="Approx_E2S_cumulative"


    def sample(self, n=100, seed=1):
        return self.gen_model.sample(n, seed=seed)
    

    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        # maximum degree between two vertices connecting edge=ij
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_stat = (deg_sum + abs(deg_diff))/2. - X
        return deg_stat
        # deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        # deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        # deg_stat = [deg_min_stat, deg_max_stat]
        # return deg_min_stat, deg_max_stat
    
    def CountDegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        idx, count = np.unique(deg_stat, return_counts = True)
        loc = np.searchsorted(idx, deg_stat)
        deg_count = np.bincount(loc.flatten(), X.flatten())
        deg_prob = deg_count.cumsum()/count.cumsum()
        
        #make a lookup table
        n = X.shape[0]
        d = X.shape[1]
        #total possible degree 0--d-1
        deg_list = np.arange(d)
        prob_list = np.zeros(d) # * 0.5
        count_list = np.zeros(d)
        count_list[0] = 1
        for i, x in enumerate(idx):
            prob_list[int(x)] = deg_prob[i]
            count_list[int(x)] = count[i]
        # prob_list = prob_list.cumsum()/count_list.cumsum()
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.count_list = count_list
        return prob_list, deg_list, count_list



class ApproxBiDegStat(ApproxModel):
    """
    model with approximated conditional edge probability of Edge+2Star graph
    The bi-variate degree is used to estimation conditional probability
    """
    def __init__(self, gen_model, n_gen=1000, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        if smooth:
            self.EstDegProb(self.Xsample.X)
        else:
            self.CountDegProb(self.Xsample.X)
        self.smooth = smooth
        self._name="Approx_BiDeg"

    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        # maximum and mininum degree between two vertices connecting edge=ij
        # remove the edge connecting edge=ij
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        deg_stat = np.array([deg_min_stat, deg_max_stat])
        return deg_stat
        # return deg_min_stat, deg_max_stat
    
    def CountDegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        deg_min_stat, deg_max_stat = deg_stat[0], deg_stat[1]
        
        deg_flat = np.array([deg_min_stat.flatten(), deg_max_stat.flatten()])
        idx, loc, count = np.unique(deg_flat, axis=1, return_counts=True, return_inverse=True)
        deg_count = np.bincount(loc, X.flatten())
        deg_prob = deg_count/count
        
        #make a lookup table
        n = X.shape[0]
        d = X.shape[1]
        #total possible degree 0--d-1
        deg_list = idx
        prob_list = np.zeros((d,d)) # * 0.5
        count_list = np.zeros((d,d))
        for i, x in enumerate(idx.T):
            prob_list[int(x[0]),int(x[1])] = deg_prob[i]
            count_list[int(x[0]),int(x[1])] = count[i]
        # prob_list = prob_list.cumsum()/count_list.cumsum()
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.count_list = count_list
        return prob_list, deg_list, count_list
    
    def cond_prob(self, X, smooth=None):
        if smooth is None:
            smooth = self.smooth 
        deg_stat = self.DegStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        if len(X.shape)==2:
            prob =  X[np.newaxis,:] * 0. 
        else:
            prob = X * 0.

        for i, deg in enumerate(deg_list.T):
            m1 = deg_stat[0] == deg[0]
            m2 = deg_stat[1] == deg[1]
            m = m1*m2
            prob[m] = prob_list[int(deg[0]), int(deg[1])]
        return prob
        
    
    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS




class ApproxETriangleStat(ApproxModel):
    """
    model with approximated conditional edge probability of Edge+Triangle graph
    """
    def __init__(self, gen_model, n_gen=100, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        self.CountProb(self.Xsample.X)
        self.smooth = smooth
        self._name="Approx_ET"

    def sample(self, n=100, seed=1):
        return self.gen_model.sample(n, seed=seed)

    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        deg_stat = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        #remove diagonal terms
        # np.apply_along_axis(np.fill_diagonal, 0, deg_stat, val=0)
        # remove the presence of edge ij
        deg_stat -= 2*X
        return deg_stat

    def NeighbourStats(self, X):
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        X2 = np.einsum("ijk, ilk -> ijl", X, X)
        return X2
        # return deg_min_stat, deg_max_stat
    
    def CountProb(self, X):
        #compute the number of common neighbours to form triangles
        neighbour_stat = self.NeighbourStats(X)
        n, m, _ = neighbour_stat.shape
        idx = (np.arange(1,m+1) + (m+1)*np.arange(m-1)[:,None] + m*m*np.arange(n)[:,None,None]).reshape(n,m,-1)
        neighbour_stat_ = neighbour_stat.ravel()[idx]
        X_ = X.ravel()[idx]
        
        
        idx, count = np.unique(neighbour_stat_, return_counts = True)
        loc = np.searchsorted(idx, neighbour_stat_)
        deg_count = np.bincount(loc.flatten(), X_.flatten())
        deg_prob = deg_count/count

        d = X.shape[1]
        deg_list = np.arange(d-1)
        prob_list = np.ones(d-1) * 0.
        count_list = np.zeros(d-1)
        for i, x in enumerate(idx):
            prob_list[int(x)] = deg_prob[i]
            count_list[int(x)] = count[i]

        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.deg_count = deg_count 
        self.count_list = count_list
        return prob_list, deg_list, count_list
    

    def cond_prob(self, X, smooth=None):
        if smooth is None:
            smooth = self.smooth 
        deg_stat = self.NeighbourStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        prob = deg_stat * 0.

        for i, deg in enumerate(deg_list):
            prob[deg_stat == deg] = prob_list[i]
        
        return prob


    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS
    

class ApproxTriangleStat(ApproxModel):
    """
    model with approximated conditional edge probability with edge and triangle count statistics
    The multi-variate vector is used to estimation conditional probability
    """
    def __init__(self, gen_model, n_gen=1000, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        if smooth:
            self.EstDegProb(self.Xsample.X)
        else:
            self.CountDegProb(self.Xsample.X)
        self.smooth = smooth
        self._name="ApproxTriangle"


    def sample(self, n=100, seed=1):
        return self.gen_model.sample(n, seed=seed)
    


    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        # maximum and mininum degree between two vertices connecting edge=ij
        # remove the edge connecting edge=ij
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        deg_stat = np.array([deg_min_stat, deg_max_stat])
        return deg_stat
        # return deg_min_stat, deg_max_stat
    
    def CountDegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        deg_min_stat, deg_max_stat = deg_stat[0], deg_stat[1]
        X2 = np.einsum("ijk, ilk -> ijl", X, X)
        tri_stat = X2 * X
        
        deg_flat = np.array([deg_min_stat.flatten(), deg_max_stat.flatten(), tri_stat.flatten()])
        idx, loc, count = np.unique(deg_flat, axis=1, return_counts=True, return_inverse=True)
        deg_count = np.bincount(loc, X.flatten())
        deg_prob = deg_count/count
        

        d = X.shape[1]
        #total possible degree 0--d-1
        deg_list = idx
        prob_list = np.zeros((2*d,d)) # * 0.5
        count_list = np.zeros((d,d))
        for i, x in enumerate(idx.T):
            prob_list[int(x[0]),int(x[1])] = deg_prob[i]
            count_list[int(x[0]),int(x[1])] = count[i]
        # prob_list = prob_list.cumsum()/count_list.cumsum()
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.count_list = count_list
        return prob_list, deg_list, count_list


    def cond_prob(self, X, smooth=None):
        if smooth is None:
            smooth = self.smooth 
        deg_stat = self.DegStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        if len(X.shape)==2:
            prob =  X[np.newaxis,:] * 0. 
        else:
            prob = X * 0.

        for i, deg in enumerate(deg_list.T):
            m1 = deg_stat[0] == deg[0]
            m2 = deg_stat[1] == deg[1]
            m = m1*m2
            prob[m] = prob_list[int(deg[0]), int(deg[1])]
        return prob
        
    
    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS
    
    



class ApproxE2STStat(ApproxModel):
    """
    model with approximated conditional edge probability with edge and triangle count statistics
    The number of common neighbour vertices is used to estimation conditional probability
    """
    def __init__(self, gen_model, n_gen=100):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        self.CountProb(self.Xsample.X)
        self._name="Approx_E2ST"
        
    def sample(self, n=100, seed=1):
        return self.gen_model.sample(n, seed=seed)
    
    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        deg_stat = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_stat -= 2*X
        return deg_stat
    
    def NeighbourStats(self, X):
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        X2 = np.einsum("ijk, ilk -> ijl", X, X)
        return X2
        # return deg_min_stat, deg_max_stat
    
    def CountProb(self, X):
        #compute the number of common neighbours to form triangles
        neighbour_stat = self.NeighbourStats(X)
        deg_stat = self.DegStats(X)
        n, m, _ = neighbour_stat.shape
        idx = (np.arange(1,m+1) + (m+1)*np.arange(m-1)[:,None] + m*m*np.arange(n)[:,None,None]).reshape(n,m,-1)
        neighbour_stat_ = neighbour_stat.ravel()[idx]
        deg_stat_ = deg_stat.ravel()[idx]
        X_ = X.ravel()[idx]
        
        
        deg_flat = np.array([deg_stat_.flatten(), neighbour_stat_.flatten()])
        idx, loc, count = np.unique(deg_flat, axis=1, return_counts=True, return_inverse=True)
        deg_count = np.bincount(loc, X_.flatten())
        deg_prob = deg_count/count
        
        d = X.shape[1]
        deg_list1 = np.arange(2*d-2)
        deg_list2 = np.arange(d-1)
        # deg_list = np.array(np.meshgrid(deg_list1, deg_list2))#.reshape([2,-1]).T
        deg_list = [deg_list1, deg_list2]
        
        prob_list = np.zeros((len(deg_list1),len(deg_list2))) # * 0.5
        count_list = np.zeros((len(deg_list1),len(deg_list2)))
        for i, x in enumerate(idx.T):
            # print(x)
            prob_list[int(x[0]),int(x[1])] = deg_prob[i]
            count_list[int(x[0]),int(x[1])] = count[i]
        # prob_list = prob_list.cumsum()/count_list.cumsum()
        
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.count_list = count_list
        
        # pred_sig = lr.predict(deg_list)
        # prob_list = utils.sigmoid(pred_sig)
        
        # self.prob_list = prob_list[:,0]
        # self.deg_list = deg_list 
        return prob_list, deg_list


    def cond_prob(self, X, fit=False):
        neighbour_stat = self.NeighbourStats(X)
        deg_stat = self.DegStats(X)
        stat = [deg_stat, neighbour_stat]

        prob_list = self.prob_list
        deg_list = self.deg_list
        
        if len(X.shape)==2:
            prob =  deg_stat[np.newaxis,:] * 0. 
        else:
            prob = X * 0.
        
        
        for i, deg in enumerate(deg_list[0]):
            m1 = stat[0] == deg
            for j, nei in enumerate(deg_list[1]):
                m2 = stat[1] == nei
                m = m1*m2
                # if m.sum()>0:
                #     print(m.sum(), deg, nei)

                prob[m] = prob_list[int(deg), int(nei)]
            
        return prob

    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS













class ApproxLocalClusterStat(ApproxModel):
    """
    model with approximated conditional edge probability with edge and triangle count statistics
    The multi-variate vector is used to estimation conditional probability
    """
    def __init__(self, gen_model, n_gen=1000, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        if smooth:
            self.EstDegProb(self.Xsample.X)
        else:
            self.CountDegProb(self.Xsample.X)
        self.smooth = smooth
        self._name="Approx_BiDeg"

    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        # maximum and mininum degree between two vertices connecting edge=ij
        # remove the edge connecting edge=ij
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        deg_stat = np.array([deg_min_stat, deg_max_stat])
        return deg_stat
        # return deg_min_stat, deg_max_stat
    
    def LocalClusterStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        deg = deg * (deg-1) /2.
        deg = deg.clip(min=0)
        # maximum and mininum degree between two vertices connecting edge=ij
        # remove the edge connecting edge=ij
        
        X2 = np.einsum("ijk, ilk -> ijl", X, X)
        X3 = np.einsum("ijk, ilk -> ijl", X2, X)
        X3d = np.einsum("ijj -> ij", X3)
        X3d = X3d.clip(min=1.)
        deg = deg/X3d
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        deg_stat = np.array([deg_min_stat, deg_max_stat])
        return deg_stat
        # return deg_min_stat, deg_max_stat
    
    def CountDegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        deg_min_stat, deg_max_stat = deg_stat[0], deg_stat[1]
        lc_stat = self.LocalClusterStats(X)
        lc_sum = lc_stat[0] + lc_stat[1]
        
        deg_flat = np.array([deg_min_stat.flatten(), deg_max_stat.flatten(), lc_sum.flatten()])
        idx, loc, count = np.unique(deg_flat, axis=1, return_counts=True, return_inverse=True)
        deg_count = np.bincount(loc, X.flatten())
        deg_prob = deg_count/count
        
        #make a lookup table
        n = X.shape[0]
        d = X.shape[1]
        #total possible degree 0--d-1
        deg_list = idx
        prob_list = np.zeros((d,d)) # * 0.5
        count_list = np.zeros((d,d))
        for i, x in enumerate(idx.T):
            prob_list[int(x[0]),int(x[1])] = deg_prob[i]
            count_list[int(x[0]),int(x[1])] = count[i]
        # prob_list = prob_list.cumsum()/count_list.cumsum()
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.count_list = count_list
        return prob_list, deg_list, count_list


    def cond_prob(self, X, smooth=None):
        if smooth is None:
            smooth = self.smooth 
        deg_stat = self.DegStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        if len(X.shape)==2:
            prob =  X[np.newaxis,:] * 0. 
        else:
            prob = X * 0.

        for i, deg in enumerate(deg_list.T):
            m1 = deg_stat[0] == deg[0]
            m2 = deg_stat[1] == deg[1]
            m = m1*m2
            prob[m] = prob_list[int(deg[0]), int(deg[1])]
        return prob
        
    
    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS
# c = nx.betweenness_centrality(G)
# for i, node in c.items():
#     print(node)
    
class ApproxDeg3Stat(ApproxModel):
    """
    model with approximated conditional edge probability of Edge+2Star graph
    The 3D vector suggested by the Reviewer (may not be the well-thought exmaple)
    """
    def __init__(self, gen_model, n_gen=1000, smooth=False):
        """
        gen_model: an implict model that are able to generate network samples
        """
        self.gen_model = gen_model
        self.n_gen = n_gen
        self.Xsample = gen_model.sample(n_gen)
        if smooth:
            self.EstDegProb(self.Xsample.X)
        else:
            self.CountDegProb(self.Xsample.X)
        self.smooth = smooth
        self._name="Approx_BiDeg"

    def DegStats(self, X):
        X = np.array(X)
        if len(X.shape)==2:
            X = X[np.newaxis, :]
        deg = X.sum(axis=1)
        # maximum and mininum degree between two vertices connecting edge=ij
        # remove the edge connecting edge=ij
        deg_sum = deg[:,:,np.newaxis] + deg[:,np.newaxis]
        deg_diff = deg[:,:,np.newaxis] - deg[:,np.newaxis]
        deg_min_stat = (deg_sum - abs(deg_diff))/2. - X
        deg_max_stat = (deg_sum + abs(deg_diff))/2. - X
        deg_stat = np.array([deg_min_stat, deg_max_stat])
        return deg_stat
        # return deg_min_stat, deg_max_stat
    
    def CountDegProb(self, X):
        #compute degree on both vertices and its corresponding edge probability
        deg_stat = self.DegStats(X)
        deg_min_stat, deg_max_stat = deg_stat[0], deg_stat[1]
        
        deg_flat = np.array([deg_min_stat.flatten(), deg_max_stat.flatten(), X.flatten()])
        idx, loc, count = np.unique(deg_flat, axis=1, return_counts=True, return_inverse=True)
        deg_count = np.bincount(loc, X.flatten())
        deg_prob = deg_count/count
        
        #make a lookup table
        n = X.shape[0]
        d = X.shape[1]
        #total possible degree 0--d-1
        deg_list = idx
        prob_list = np.zeros((d,d)) # * 0.5
        count_list = np.zeros((d,d))
        for i, x in enumerate(idx.T):
            prob_list[int(x[0]),int(x[1])] = deg_prob[i]
            count_list[int(x[0]),int(x[1])] = count[i]
        # prob_list = prob_list.cumsum()/count_list.cumsum()
        self.prob_list = prob_list 
        self.deg_list = deg_list 
        self.count_list = count_list
        return prob_list, deg_list, count_list
    
    def cond_prob(self, X, smooth=None):
        if smooth is None:
            smooth = self.smooth 
        deg_stat = self.DegStats(X)
        prob_list = self.prob_list
        deg_list = self.deg_list
        if len(X.shape)==2:
            prob =  X[np.newaxis,:] * 0. 
        else:
            prob = X * 0.

        for i, deg in enumerate(deg_list.T):
            m1 = deg_stat[0] == deg[0]
            m2 = deg_stat[1] == deg[1]
            m = m1*m2
            prob[m] = prob_list[int(deg[0]), int(deg[1])]
        return prob
        
    
    def get_datasource(self):
        ###the generator here is used as datasource
        DS = self.gen_model
        return DS

