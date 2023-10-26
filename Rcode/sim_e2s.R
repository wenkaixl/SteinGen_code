setwd("/home/wenkaix/Workspace/SteinGen/Rcode")
library(network)
library(ergm)

# source("../Rcode/sim_ergm.R")

d.list = c(100, 500, 1000, 1500, 2000)
N = 20

for (di in 1:5){
  d = d.list[di]
  print(d)
  un=network(d, directed = FALSE)
  model0<- un ~ edges +kstar(2)
  coef = c(-1.5, 1.0/d)
  start_time <- Sys.time()
  g.sim  <- simulate(model0, nsim=N, coef=coef, 
                     control=control.simulate(MCMC.burnin=d*(100), MCMC.batch=20))
  
  # generate the network adjacency matrices  
  g.adj = c()
  for (ii in 1:N){
    g.adj= c(g.adj, list(g.sim[[ii]][,]))
  }
  end_time <- Sys.time()
  print(end_time - start_time)
  
  filename = paste(paste("E2S100_b1_15_n",d, sep = ""),".rds", sep="")
  saveRDS(g.adj, file=filename)
  rm(g.sim)
  rm(g.adj)
  gc()
  print("done")
}


# 
# 
# for (ii in 1:N-1){
#   g1 = as.matrix(g.sim[[ii+1]])
#   g2 = as.matrix(g.sim[[ii+2]])
#   print(sum(abs(g1-g2)))
# }
