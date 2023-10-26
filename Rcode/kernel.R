
library(ergm)
library(network)
library(igraph)
library(graphkernels)


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


## Geometric Random Walk Kernel
compute.grw.kernel=function(X, level=3, diag=1, normalize=TRUE){
  n = length(X)
  P = compute.transition.list(X)
  kernel.matrix = CalculateGeometricRandomWalkKernel(P, level)
  rm(P)
  K = kernel.matrix[1:n,1:n] + kernel.matrix[n+1,n+1]
  K.vec = kernel.matrix[1:n,n+1]
  K = K + outer(K.vec, K.vec, function(x,y)x+y)
  if(normalize)K = compute.normalized(K)
  if(diag==0)diag(K)<-0
  K
}


## Shortest Path Kernel
compute.sp.kernel=function(X, level=1, diag=1, normalize=TRUE){
  n = length(X)
  P = compute.transition.list(X)
  kernel.matrix = CalculateShortestPathKernel(P)
  rm(P)
  K = kernel.matrix[1:n,1:n] + kernel.matrix[n+1,n+1]
  K.vec = kernel.matrix[1:n,n+1]
  K = K + outer(K.vec, K.vec, function(x,y)x+y)
  if(normalize)K = compute.normalized(K)
  if(diag==0)diag(K)<-0
  K
}


## Vertex-Edge Histogram Kernel
compute.veh.kernel=function(X, level=0.1, diag=1, normalize=TRUE){
  n = length(X)
  P = compute.transition.list(X)
  kernel.matrix = CalculateVertexEdgeHistGaussKernel(P,level)
  rm(P)
  K = kernel.matrix[1:n,1:n] + kernel.matrix[n+1,n+1]
  K.vec = kernel.matrix[1:n,n+1]
  K = K + outer(K.vec, K.vec, function(x,y)x+y)
  if(normalize)K = compute.normalized(K)
  if(diag==0)diag(K)<-0
  K
}


# Various special kernels

compute.linear.kernel=function(X,normalize = TRUE,diag=1){
  n = length(X)
  K = matrix(1,ncol=n,nrow = n) 
  if(diag==0)diag(K)<-0
  K
}

compute.constant.kernel=function(X,normalize = TRUE,diag=1){
  n = length(X)
  K = matrix(0,ncol=n,nrow = n) + 1
  if(diag==0)diag(K)<-0
  K
}

compute.flipsign.kernel=function(X,normalize = TRUE,diag=1){
  n = length(X)
  S = matrix(X, byrow =TRUE)*2 - 1
  K = S%*%t(S)
  if(diag==0)diag(K)<-0
  K
}


