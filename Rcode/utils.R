library(ergm)
library(network)
library(graphkernels)
library(igraph)
library(MASS)

source("../Rcode/kernel.R")


# kernels and related methods
compute.transition.list = function(X){
  P=list()
  for (w in 1:length(X)){
    x = X
    x[w] = abs(1-X[w])
    G = graph_from_adjacency_matrix(x)
    P[[w]] = G
  }
  P[[length(X)+1]] = graph_from_adjacency_matrix(X)
  P
}


compute.sampled.list = function(X, sample.index){
  P=list()
  l = length(sample.index)
  for (w in 1:l){
    x = X
    x[sample.index[w]] = abs(1 - X[sample.index[w]])
    G = graph_from_adjacency_matrix(x)
    P[[w]] = G
  }
  P[[l+1]] = graph_from_adjacency_matrix(X)
  P
}


compute.normalized = function(K){
  V = diag(K)
  D.mat = diag(1.0/sqrt(V))
  D.mat%*%K%*%D.mat
}


## Weisfeiler_Lehman Graph Kernel
compute.wl.kernel=function(X, level=3, diag=1, normalize=TRUE){
  n = length(X)
  P = compute.transition.list(X)
  kernel.matrix = CalculateWLKernel(P, level)
  rm(P)
  K = kernel.matrix[1:n,1:n] + kernel.matrix[n+1,n+1]
  K.vec = kernel.matrix[1:n,n+1]
  K = K + outer(K.vec, K.vec, function(x,y)x+y)
  if(normalize)K = compute.normalized(K)
  if(diag==0)diag(K)<-0
  K
}


# GKSS methods


# Using vvRKHS
GKSS_condition=function(X, q_X, kernel=compute.wl.kernel,diagonal=1)
{
  S=matrix(X,byrow=TRUE)
  n.sim=length(S)
  S.t =  q_X
  S.t.vec=abs(S - matrix(S.t, byrow=TRUE))
  S.mat = S.t.vec %*% t(S.t.vec)
  Ky.mat = (S*2-1)%*%t(S*2-1) 
  
    
  K = kernel(X, diag=diagonal)
  J.kernel = S.mat * Ky.mat * K
  
  W=rep(1/n.sim,n.sim)
  J.kernel.out=J.kernel
  if(diagonal==0)diag(J.kernel)=rep(0,n.sim)
  stats.value=n.sim*t(W)%*%J.kernel%*%W #* sqrt(v.kernel)
  #Return:
  #stats.value: n times KSD
  #J.kernel: J kernel matrix for wild bootstrapt 
  list(stats.value=stats.value, J.kernel=J.kernel.out, K=K, S=S.mat)
}



# Re-Sample algorithms 
GKSS_sampled=function(t_fun, X, sample.index, g.kernel=CalculateWLKernel,level=3, diagonal=1, normalize=TRUE, v.scale=TRUE)
{
  S=matrix(X,byrow=TRUE)[sample.index]
  n.sim=length(S)
  S.t =  t_fun(X)[sample.index]
  S.t.vec=abs(S - S.t)
  S.mat = S.t.vec %*% t(S.t.vec)
  Ky.mat = (S*2-1)%*%t(S*2-1) 
  
  
  n = length(sample.index)
  P = compute.sampled.list(X, sample.index)
  kernel.matrix = g.kernel(P, level)
  K = kernel.matrix[1:n,1:n] + kernel.matrix[n+1,n+1]
  K.vec = kernel.matrix[1:n,n+1]
  K = K + outer(K.vec, K.vec, function(x,y)x+y)
  if(normalize)K = compute.normalized(K)
  
  J.kernel = S.mat * Ky.mat * K
  
  W=rep(1/n,n)
  J.kernel.out=J.kernel
  if(diagonal==0)diag(J.kernel)=rep(0,n.sim)
  v.kernel = var(S)
  stats.value=n*t(W)%*%J.kernel%*%W 
  if(v.scale)nKSD = stats.value* sqrt(v.kernel)
  #Return:
  #stats.value: n times KSD
  #J.kernel: J kernel matrix for wild bootstrapt 
  list(stats.value=stats.value,J.kernel=J.kernel.out, K=K, S=S.mat)
}


GOF_deg = function(t_coef, X)
{ Xnet = network(X, directed=FALSE)
  gest = ergm(Xnet ~ edges, directed=FALSE)
  gnull <- gof(gest, GOF=~degree, coef=t_coef)
  obs = gnull$obs.deg
}


generate.degree.stats=function(t_fun, X, sample.index, mean=0, kernel=CalculateWLKernel,level=3, diagonal=1, normalize=TRUE, v.scale=TRUE)
{
  Deg = colSums(X)
  n = dim(X)[1]
  stats.value = mean((Deg-mean)^2)
  list(stats.value=stats.value)
}


generate.graphical.stats.degree=function(t_coef, X, sample.index, kernel=CalculateWLKernel,level=3, diagonal=1, normalize=TRUE, v.scale=TRUE)
{
  #X here is an ergm
  gest = ergm(X~ edges + kstar(2) + triangles, estimate = "MPLE")
  # gnull <- gof(gest, GOF=~distance + espartners + degree + dspartners+
  #                triadcensus, coef=t_coef)
  gnull <- gof(gest, GOF=~degree, coef=t_coef)
  obs = gnull$obs.deg
  obs = obs/sum(obs)
  sims = colMeans(gnull$sim.deg)
  sims = sims/sum(sims)
  stats.value = 0.5*sum(abs(obs-sims))
  list(stats.value=stats.value)
}

generate.graphical.stats.distance=function(t_coef, X, sample.index, kernel=CalculateWLKernel,level=3, diagonal=1, normalize=TRUE, v.scale=TRUE)
{
  #X here is an ergm
  gest = ergm(X~ edges + kstar(2) + triangles, estimate = "MPLE")
  # gnull <- gof(gest, GOF=~distance + espartners + degree + dspartners+
  #                triadcensus, coef=t_coef)
  gnull <- gof(gest, GOF=~distance, coef=t_coef)
  obs = gnull$obs.dist
  obs = obs/sum(obs)
  sims = colMeans(gnull$sim.dist)
  sims = sims/sum(sims)
  stats.value = 0.5*sum(abs(obs-sims))
  list(stats.value=stats.value)
}

generate.graphical.stats.espart=function(t_coef, X, sample.index, kernel=CalculateWLKernel,level=3, diagonal=1, normalize=TRUE, v.scale=TRUE)
{
  #X here is an ergm
  gest = ergm(X~ edges + kstar(2) + triangles, estimate = "CD")
  # gnull <- gof(gest, GOF=~distance + espartners + degree + dspartners+
  #                triadcensus, coef=t_coef)
  gnull <- gof(gest, GOF=~espartners, coef=t_coef)
  obs = gnull$obs.espart
  obs = obs/sum(obs)
  sims = colMeans(gnull$sim.espart)
  sims = sims/sum(sims)
  stats.value = 0.5*sum(abs(obs-sims))
  list(stats.value=stats.value)
}

generate.MD.degree=function(t_coef, X)
{
  gest = ergm(X~ edges + kstar(2) + triangles, estimate = "MPLE")
  gnull <- gof(gest, GOF=~degree, coef=t_coef)
  Ax = gnull$obs.deg
  sim.null = gnull$sim.deg
  mu = colMeans(sim.null)
  Sig = cov(sim.null)
  stats.value = t(Ax-mu) %*% ginv(Sig) %*%(Ax-mu) 
  list(stats.value=stats.value)
}

###perform MC-based test####

perform.test = function(generate.method, model.h1, coef.h1, n, sim.h0){
  g.sim.data <- simulate(model.h1, nsim=n,
                         coef=coef.h1,control=control.simulate(MCMC.burnin=1000, MCMC.interval=500))
  N = dim(sim.h0)[1]
  pval.list = matrix(0,n)
  test.stats.list = matrix(0,n)
  for (i in 1:n){
    X = g.sim.data[[i]][,]
    
    ##method
    test.out=generate.method(t_fun, X)
    test.stats = test.out[['stats.value']]; 
    # print(paste("Test stats:",test.stats))
    pval = mean(rep(test.stats, N)<sim.h0)
    # print(pval)
    pval.list[i] = pval
    test.stats.list[i] = test.stats
    if(i%%100==0)print(paste("Iter",i, (pval)))
  }
  list(pvalue=pval.list, stats=test.stats.list)
}

perform.test.sample = function(generate.method, model.h1, coef.h1, n, B, sim.h0){
  g.sim.data <- simulate(model.h1, nsim=n,
                         coef=coef.h1,control=control.simulate(MCMC.burnin=1000, MCMC.interval=50))
  N = dim(sim.h0)[1]
  pval.list = matrix(0,n)
  test.stats.list = matrix(0,n)
  d = dim(g.sim.data[[1]][,])[1]
  idx = sample.int(d^2, size = B, replace = TRUE)
  for (i in 1:n){
    X = g.sim.data[[i]][,]
    ##method
    test.out=generate.method(t_fun, X, idx)
    test.stats = test.out[['stats.value']]; 
    # print(paste("Test stats:",test.stats))
    pval = mean(rep(test.stats, N)<sim.h0)
    # print(pval)
    pval.list[i] = pval
    test.stats.list[i] = test.stats
    if(i%%100==0)print(paste("Iter",i, (pval)))
  }
  list(pvalue=pval.list, stats=test.stats.list)
}


perform.test.graphical = function(generate.method, model.h1, coef.h1, n, B, sim.h0, coef.h0=c(-2, -0.0, 0.01)){
  g.sim.data <- simulate(model.h1, nsim=n,
                         coef=coef.h1,control=control.simulate(MCMC.burnin=1000, MCMC.interval=500))
  N = dim(sim.h0)[1]
  pval.list = matrix(0,n)
  test.stats.list = matrix(0,n)
  d = dim(g.sim.data[[1]][,])[1]
  idx = sample.int(d^2, size = B, replace = TRUE)
  for (i in 1:n){
    X = g.sim.data[[i]]
    
    ##method
    test.out=generate.method(coef.h0, X, idx)
    test.stats = test.out[['stats.value']]; 
    # print(paste("Test stats:",test.stats))
    pval = mean(rep(test.stats, N)<sim.h0)
    # print(pval)
    pval.list[i] = pval
    test.stats.list[i] = test.stats
    if(i%%100==0)print(paste("Iter",i, (pval)))
  }
  list(pvalue=pval.list, stats=test.stats.list)
}

perform.test.size = function(generate.method, model.h1, coef.h1, n, sim.h0, B.list){
  g.sim.data <- simulate(model.h1, nsim=n,
                         coef=coef.h1,control=control.simulate(MCMC.burnin=1000, MCMC.interval=500))
  N = dim(sim.h0)[2]
  l = length(B.list)
  pval.list = matrix(0,l,n)
  test.stats.list = matrix(0,l,n)
  for (i in 1:n){
    X = g.sim.data[[i]][,]
    
    ##method
    test.out=generate.method(t_fun, X)
    J = test.out[['J.kernel']]
    for (w in 1:l){
      B = B.list[w]
      idx = sample.int(d^2, size = B, replace = TRUE)
      Mat = J[idx, idx]
      test.stats = mean(Mat)
      pval = mean(rep(test.stats, N)<sim.h0[w,])
      pval.list[w,i] = pval
      test.stats.list[w,i] = test.stats
      if(i%%100==0)print(paste("Iter",i, (pval)))
    }
  }
  list(pvalue=pval.list, stats=test.stats.list)
}

### compute test statistics###
MDdegree=function(t_coef, X)
{ 
  un = network(X, directed=FALSE)
  gest = ergm(un~ edges, estimate = "MPLE")
  gnull <- gof(gest, GOF=~degree, coef=t_coef)
  Ax = gnull$obs.deg
  sim.null = gnull$sim.deg
  mu = colMeans(sim.null)
  Sig = cov(sim.null)
  stats.value = t(Ax-mu) %*% ginv(Sig) %*%(Ax-mu) 
  list(stats.value=stats.value)
}

Param_degree=function(X, estimate = "MPLE")
{ 
  un = network(X, directed=FALSE)
  gest = ergm(un~ edges, estimate = estimate)
  stats.value = gest$coef[[1]]
  list(stats.value=stats.value)
}



# data(florentine)
#flomarriage
# X = flomarriage
#t_coef = c(0,0,0)
#gest = ergm(X~ edges + kstar(2) + triangles, estimate = "CD")
#gnull <- gof(gest, GOF=~distance + espartners + degree + dspartners+
#                triadcensus, coef=t_coef)
#gnull <- gof(gest, GOF=~degree, coef=t_coef)
#obs = gnull$obs.deg
#obs = obs/sum(obs)
#sims = colMeans(gnull$sim.deg)


