
#############################################
### Protein-Protein Interaction Network #####
#############################################
library("igraph")
library("igraphdata")
data(yeast)
g.ppi <- yeast

setwd("~/Workspace/SteinGen/Rcode/")
load(file="virusppi.rda")
virusppi

data_dir <- file.path("inst", "extdata", "VRPINS")

load_virus_data <- function(filename) {
  read_simple_graph(file = file.path(data_dir, filename), format = "ncol")
}

virusppi <- list(
  EBV = load_virus_data("EBV.txt"),
  ECL = load_virus_data("ECL.txt"),
  `HSV-1` = load_virus_data("HSV-1.txt"),
  KSHV = load_virus_data("KSHV.txt"),
  VZV = load_virus_data("VZV.txt")
)

devtools::use_data(virusppi, overwrite = TRUE)
