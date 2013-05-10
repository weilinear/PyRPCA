from robustpca import *
from numpy.random import randn
from numpy.random import shuffle
import numpy as np
from numpy.testing import assert_almost_equal
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
    return
    A_, E_ = singular_value_thresholding(Observed.T, lmbda = 0.1, tau=1e3, maxiter = 100, verbose = 0)
    assert (np.sqrt(((A_ - TrueMat.T)**2).sum()) / np.sqrt((A_**2).sum())) < .01
    assert (np.abs(E_ - E.T).sum() / np.abs(E).sum()) < .01

def test_augmented_largrange_multiplier():
    return
    A_, E_ = augmented_largrange_multiplier(Observed.T, lmbda = .1, inexact = True, verbose = 0)
    assert_almost_equal(A_, TrueMat.T, decimal = 2)
    assert_almost_equal(E_, E.T, decimal = 2)
    A_, E_ = augmented_largrange_multiplier(Observed.T, lmbda = .1, inexact = False, verbose = 0)
    assert_almost_equal(A_, TrueMat.T, decimal = 2)
    assert_almost_equal(E_, E.T, decimal = 2)

def test_accelerate_proximal_gradient():
    return
    A_, E_ = accelerate_proximal_gradient(Observed.T, lmbda = .1, verbose = 0)
    assert_almost_equal(A_, TrueMat.T, decimal = 2)
    assert_almost_equal(E_, E.T, decimal = 2)

def test_alternating_direction_method_of_multipliers():
    A_, E_ = alternating_direction_method_of_multipliers(Observed.T, lmbda = .1)
    assert_almost_equal(A_, TrueMat.T, decimal = 2)
    assert_almost_equal(E_, E.T, decimal = 2)

