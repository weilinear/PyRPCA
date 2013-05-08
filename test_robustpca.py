from robustpca import *
from numpy.random import randn
from numpy.random import shuffle

# initalize a basis of rank 10
Basis = randn(5, 100)
W     = randn(2000, 5)
TrueMat = np.dot(W, Basis)
# initalize a sparse matrix
sparse_ratio = 0.01
sparse_mag  = 10
E     = randn(TrueMat.size) * sparse_mag
idx   = np.arange(E.size); shuffle(idx)
E[idx[int(sparse_ratio * E.size):]] = 0
E     = E.reshape(TrueMat.shape)
# calculate the observation
Observed = TrueMat + E # 1000 x 100

def test_singular_value_thresholding():
    singular_value_thresholding(Observed.T, lmbda = 0.01)
    
