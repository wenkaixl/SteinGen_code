
library(network)
library(ergm)
library(ergm.multi)

# construct the networks
construct_er_model = function(d){
  un=network(d, directed = FALSE)
  model0<- un ~ edges
}


construct_e2s_model = function(d){
  un=network(d, directed = FALSE)
  model0<- un ~ edges +kstar(2)
}

construct_e3s_model = function(d){
  un=network(d, directed = FALSE)
  model0<- un ~ edges +kstar(3)
}

construct_eks_model = function(d, k=4){
  un=network(d, directed = FALSE)
  model0<- un ~ edges +kstar(k)
}

construct_et_model = function(d){
  un=network(d, directed = FALSE)
  model0<- un ~ edges + triangles
}


construct_e2st_model = function(d){
  un=network(d, directed = FALSE)
  model0<- un ~ edges + kstar(2) + triangles
}



# generate the network adjacency matrices  


gen_ergm = function(d=20, N=500, construct=construct_er_model, coef=c(0)){
    # d: size of the network
    # N: number of network samples
    # construct: the method to construct ergm model
    #coef: coeficient for network statistics

    model0 = construct(d)
    g.sim  <- simulate(model0, nsim=N, coef=coef, 
                        control=control.simulate(MCMC.burnin=d*(d)))
    
    if (N==1){
        g.adj = g.sim[,]}
    else{
        g.adj = c()
        for (ii in 1:N){
                g.adj= c(g.adj, list(g.sim[[ii]][,]))
            }
    }
    g.adj
}



gen_ergm_list = function(d=20, N=500, construct=construct_er_model, coef=c(0), return_list=FALSE){
  # d: size of the network
  # N: number of network samples
  # construct: the method to construct ergm model
  #coef: coeficient for network statistics
  
  model0 = construct(d)
  g.sim  <- simulate(model0, nsim=N, coef=coef, 
                     control=control.simulate(MCMC.burnin=10+d*log(d)))
  
  if (N==1){
    g.adj = g.sim[,]}
  else{
    g.adj = c()
    for (ii in 1:N){
      g.adj= c(g.adj, list(g.sim[[ii]][,]))
    }
  }
  list(g.sim, g.adj)
}


estimate_er = function(X, estimate_method="MLE"){
  un = network(X, directed=FALSE)
  gest = ergm(un~ edges, estimate = estimate_method)
  gest["coefficients"]$coefficients
}


estimate_e2s = function(X, estimate_method="MLE"){
  un = network(X, directed=FALSE)
  gest = ergm(un~ edges + kstar(2), estimate = estimate_method)
  gest["coefficients"]$coefficients
}


estimate_e3s = function(X, estimate_method="MLE"){
  un = network(X, directed=FALSE)
  gest = ergm(un~ edges + kstar(3), estimate = estimate_method)
  gest["coefficients"]$coefficients
}

estimate_eks = function(X, estimate_method="MLE", k=4){
  un = network(X, directed=FALSE)
  gest = ergm(un~ edges + kstar(k), estimate = estimate_method)
  gest["coefficients"]$coefficients
}



estimate_e2st = function(X, estimate_method="MLE"){
  un = network(X, directed = FALSE)
  gest = ergm(un~ edges + kstar(2) + triangles, estimate = estimate_method)
  gest["coefficients"]$coefficients
}

estimate_et = function(X, estimate_method="MLE"){
  un = network(X, directed = FALSE)
  gest = ergm(un~ edges + triangles, estimate = estimate_method)
  gest["coefficients"]$coefficients
}


estimate_e2st_san = function(X, stats){
  un = network(X, directed = FALSE)
  gest = san(un~ edges + kstar(2) + triangles, target.stats = stats)
  #gest["coefficients"]$coefficients
  gest[,]
}


sim_teenager = function(){
library("RSiena")
g.teen <- s501
X = g.teen[,]
}



sim_florentine = function(){
data(florentine)
g.flo <- flomarriage
X = g.flo[,]
}


sim_lazega = function(){
library("sand")
X <- as_adjacency_matrix(lazega)
X = as.matrix(X)
}



sim_ppi = function(){
  library("igraph")
  library("igraphdata")
  data(yeast)
  X <- as_adjacency_matrix(yeast)
  X = as.matrix(X)
}


sim_virus = function(i=1){
  setwd("~/Workspace/SteinGen/Rcode/")
  load(file="virusppi.rda")
  v_name = names(virusppi)
  if (i==1){
    ig = virusppi$EBV
  }
  else if (i==2){
    ig = virusppi$VZV
  }
  else if (i==3){
    ig = virusppi$`HSV-1`
  }
  else if (i==4){
    ig = virusppi$KSHV
  }
  else if (i==5){
    ig = virusppi$ECL
  }
  
  X <- as_adjacency_matrix(ig)
  X = as.matrix(X)
}



# X = gen_ergm(20, 3)

network_list_from_adj = function(X){
  g.sim = list()
  N = length(X)
  for (ii in 1:N){
    g.sim = network.list(g.sim, network(X[[ii]]))
  }
  g.sim
}

estimate_e2s_multi = function(g.sim, estimate_method="MLE"){
  g.simN <- Networks(g.sim, directed=FALSE)
  gest = ergm(g.simN ~ N(~edges+kstar(2)), estimate=estimate_method)
  gest["coefficients"]$coefficients
}

estimate_e2st_multi = function(g.sim, estimate_method="MLE"){
  g.simN <- Networks(g.sim, directed=FALSE)
  gest = ergm(g.simN ~ N(~edges+kstar(2)+triangles), estimate=estimate_method)
  gest["coefficients"]$coefficients
}


estimate_er_multi = function(g.sim, estimate_method="MLE"){
  g.simN <- Networks(g.sim, directed=FALSE)
  gest = ergm(g.simN ~ N(~edges), estimate=estimate_method)
  gest["coefficients"]$coefficients
}


estimate_et_multi = function(g.sim, estimate_method="MLE"){
  g.simN <- Networks(g.sim, directed=FALSE)
  gest = ergm(g.simN ~ N(~edges+triangles), estimate=estimate_method)
  gest["coefficients"]$coefficients
}

# d=20
# un = network(d, directed=FALSE)
# model = un ~ edges + kstar(2) + triangles
# coef3 =c(-2, 0.01, 0.01)
# g.sim = simulate(model, nsim=10, coef=coef3, control=control.simulate(MCMC.burnin = 100+10*d*log(d), MCMC.interval = d))
# 
# Networks(g.simN,network(X[[1]]))
# 
# g.simN <- Networks(network(X[[ii]]),network(X[[2]]))
# ergm(g.simN ~ N(~edges+kstar(2)+triangles), estimate="MLE")
# 
# class(g.simN)
# 






# model0 = construct_e2s_model(20)
# coef=c(-2,0.1)
# d = 20
# N = 1
# simulate(model0, nsim=N, coef=coef, control=control.simulate(MCMC.burnin=100+10*d, MCMC.interval=d))
# control.ergm(init = c(-2,1))


# data(florentine)
# ergm.godfather(flomarriage~edges+absdiff("wealth")+triangles,
#                changes=list(cbind(1:2,2:3),
#                             cbind(3,5),
#                             cbind(3,5),
#                             cbind(1:2,2:3)),
#                stats.start=FALSE
#                )
