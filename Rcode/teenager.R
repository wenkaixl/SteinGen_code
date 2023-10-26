
#####################################
### Teenager friendship network #####
#####################################
library("RSiena")
library("ergm")
g.teen <- s501
X = g.teen
p.teen=sum(X)/(25*49)
a.star.teen = logit(p.teen)
p.teen
a.star.teen

d=50
N=200
un.teen=network(d, directed = FALSE)
model0.teen<- un.teen ~ edges
sim.teen <- matrix(1,N)
g.sim.teen <- simulate(model0.teen, nsim=N,
                  coef=a.star.teen,control=control.simulate(MCMC.burnin=1000+10*d, MCMC.interval=d))

coef.h0 = c(a.star.teen,0,0)
idx =  sample.int(d^2, size = 300, replace = TRUE)
for (ii in 1:N){
  X.sim = g.sim.teen[[ii]][,]
  method.out=generate.one.GKSS.sampled(t_fun, X.sim, idx)
  sim.teen[ii] = method.out[['stats.value']]
  if(ii%%20==0)print(paste("Iter",ii, (sim[ii])))
}

res.teen = generate.one.GKSS.sampled(t_fun, X,idx)
stats.teen = res.teen[['stats.value']]#/sqrt(50)
K.teen = res.teen[['J.kernel']]
stats.teen

mean(rep(stats.teen,N) < sim.teen[1:N])
