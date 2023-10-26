library(ergm)
library(network)
library(igraph)

sim_ppi = function(){
  library("igraph")
  library("igraphdata")
  data(yeast)
  X <- as_adjacency_matrix(yeast)
  X = as.matrix(X)
}


sim_virus = function(i=1){
  # setwd("~/Workspace/SteinGen/Rcode/")
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

