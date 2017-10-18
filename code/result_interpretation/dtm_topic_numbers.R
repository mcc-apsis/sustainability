library(RcppCNPy)
library(NMF)

x <- npyLoad("data/338_B.npy")

estim.r <- nmf(x, 10:11, seed = "nndsvd")
