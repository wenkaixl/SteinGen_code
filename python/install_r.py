#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# install packages in case needed
# import rpy2's package module
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
utils = rpackages.importr('utils')

packnames = ["igraph", "graphkernels", "ergm", "ergm.multi"]
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# utils = importr('utils')
# utils.install_packages(StrVector("ergm"))

