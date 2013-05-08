from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

try:
    from pypropack import svdp
    svd = lambda X, k : svdp(X, k, 'L', kmax = max(10, 5 * k))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
except:
    from scipy.linalg import svd as svd_
    def svd(X, k):
        U, S, V = svd_(X)
        return U[:,:k], S[:k], V[:k,:]

# The problem solved is
#                   min  : tau * (|A|_* + \lmbda |E|_1) + .5 * |(A,E)|_F^2
#              subject to: A + E = D
def _monitor(A, E, D):
    diags = svd(A, min(A.shape))[1]
    print "|A|_*" , np.abs(diags).sum()
    print "|A|_0" , (np.abs(diags) > 1e-6).sum()
    print "|E|_1" , np.abs(E).sum()
    print "|D-A-E|_F", ((D - A - E) ** 2).sum()
    
def _pos(A):
    return A * (A > 0)

def singular_value_thresholding(D, maxiter = 25000, lmbda = 1.0, tau = 1e4, delta = .9, verbose = 2):
    """
    Singular Value Thresholding
    """
    # initialization
    _matshape = D.shape
    Y = np.zeros(shape = _matshape)
    A = np.zeros(shape = _matshape)
    E = np.zeros(shape = _matshape)
    rankA = 0
    for iter in range(maxiter):
        U, S, V = svd(Y, rankA+1)
        print S
        A = np.dot(np.dot(U, np.diag(_pos(S - tau))), V)
        E = np.sign(Y) * _pos(np.abs(Y) - lmbda * tau)
        M = D - A - E
        rankA = (S > tau).sum()
        Y = Y + delta * M
        if verbose >= 2:
            _monitor(A, E, D)
    return A, E    

def accelerate_proximal_gradient():
    """
    Accelerated Proximal Gradient (Partial SVD Version)
    """
    pass

def dual_method():
    """
    Dual Method
    """
    pass

def augmented_largrange_multiplier(D, lmbda, tol = 1e-7, maxiter = 25000):
    """
    Augmented Lagrange Multiplier
    """
    Y = sign(D)
    norm_two = svd(Y, 1)
    norm_inf = np.abs(Y).max() / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A_hat = np.zeros(Y.shape)
    E_hat = np.zeros(Y.shape)
    dnorm = np.sqrt((D ** 2).sum())
    tolProj = 1e-6 * dnorm
    total_svd = 0
    mu = .5/norm_Two
    rho = 6

    sv = 5
    svp = sv

    for iter in range(maxiter):
        tempT = D - A_hat + (1/mu) * Y
        temp_E = (temp_T - lmbda/mu).max(axis=0) + (temp_T + lmbda / mu).max(axis=0)
        
    
    pass

class RobustPCA(BaseEstimator, TransformerMixin):
    """
    Robust PCA
    """
    def __init__(self, alpha = .1, copy=True, method='svt'):
        """
        """
        pass

    def tranform():
        """
        Tranform
        """


