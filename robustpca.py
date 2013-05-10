from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import scipy.sparse as sp

try:
    from pypropack import svdp
    raise ValueError
    svd = lambda X, k: svdp(X, k, 'L', kmax=max(100, 10 * k))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
except:
    from scipy.linalg import svd as svd_

    def svd(X, k=-1):
        U, S, V = svd_(X, full_matrices=False)
        if k < 0:
            return U, S, V
        else:
            return U[:, :k], S[:k], V[:k, :]

# The problem solved is
#                   min  : tau * (|A|_* + \lmbda |E|_1) + .5 * |(A,E)|_F^2
#              subject to: A + E = D


def _monitor(A, E, D, lmbda=0.1):
    diags = svd(A, min(A.shape))[1]
    print "|A|_*", np.abs(diags).sum()
    print "|A|_0", (np.abs(diags) > 1e-6).sum()
    print "|E|_1", np.abs(D - A).sum()
    print "|D-A-E|_F", _fro(D - A - E)
    return np.abs(diags).sum() + lmbda * np.abs(D - A).sum()


def _pos(A):
    return A * (A > 0)


def _fro(A):
    return np.sqrt((A * A).sum())


def singular_value_thresholding(D, maxiter=25000, lmbda=1.0, tau=1e4, delta=.9, verbose=2):
    """
    Singular Value Thresholding
    """
    # initialization
    _matshape = D.shape
    primal_tol = 1e-5
    Y = np.zeros(shape=_matshape)
    A = np.zeros(shape=_matshape)
    E = np.zeros(shape=_matshape)
    rankA = 0
    obj = []
    for iter in range(maxiter):
        U, S, V = svd(Y, rankA+1)
        A = np.dot(np.dot(U, np.diag(_pos(S - tau))), V)
        E = np.sign(Y) * _pos(np.abs(Y) - lmbda * tau)
        M = D - A - E
        rankA = (S > tau).sum()
        Y = Y + delta * M
        if verbose >= 2:
            obj.append(_monitor(A, E, D))
        if _fro(D-A-E)/_fro(D) < primal_tol:
            if verbose >= 2:
                print "Converged at iter %d"%iter
            break
    if verbose >= 2:
        return A, E, obj
    else:
        return A, E


def accelerate_proximal_gradient(D, lmbda, maxiter=25000, tol=1e-7,
                                 continuation=True,
                                 eta=.9, mu=1e-3, verbose=2):
    """
    Accelerated Proximal Gradient (Partial SVD Version)
    """
    obj = []
    m, n = D.shape

    t_k = 1.
    tk_old = 1.
    tau_0 = 2.

    A_old = np.zeros(D.shape)
    E_old = np.zeros(D.shape)
    A = np.zeros(D.shape)
    E = np.zeros(D.shape)

    # This comes from the code
    if continuation:
        mu_0 = svd(D, 1)[1]
        mu_k = mu_0;
        mu_bar = 1e-9 * mu_0
    else:
        mu_k = mu;

    tau_k = tau_0;
    converged = False;
    sv = 5.;

    for iter in range(maxiter):
        YA = A + ((tk_old - 1)/t_k)*(A-A_old);
        YE = E + ((tk_old - 1)/t_k)*(E-E_old);

        A_old = YA - (1/tau_k)*(YA+YE-D);
        E_old = YE - (1/tau_k)*(YA+YE-D);

        U, S, V = svd(A_old);

        svp = (S > mu_k/tau_k).sum();
        # this line to update the number of singular values comes from the code
        if svp < sv:
            sv = min(svp + 1, n);
        else:
            sv = min(svp + round(0.05*n), n);

        A_new = np.dot(
            np.dot(U[:, :svp], np.diag(S[:svp] - mu_k/tau_k)), V[:svp, :])
        E_new = np.sign(E_old) * _pos(np.abs(E_old) - lmbda * mu_k / tau_k);

        t_kp1 = 0.5*(1+np.sqrt(1+4*t_k*t_k));

        A_old = A_new + E_new - YA - YE;
        YA = tau_k*(YA-A_new) + A_old;
        YE = tau_k*(YE-E_new) + A_old;

        s1 = np.sqrt((YA**2).sum()+(YE**2).sum())
        s2 = np.sqrt((A_new**2).sum()+(E_new**2).sum());

        if s1 / (tau_k*max(1, s2)) <= tol and iter > 10:
            break;

        if continuation:
            mu_k = max(0.9*mu_k, mu_bar);

        tk_old = t_k;
        t_k = t_kp1;
        A_old = A;
        E_old = E;
        A = A_new;
        E = E_new;

        if verbose >= 2:
            obj.append(_monitor(A, E, D))
        if (not converged) and iter >= maxiter:
            print 'Maximum iterations reached'
            converged = True;
    if verbose >= 2:
        return A, E, obj
    else:
        return A, E


def dual_method():
    """
    Dual Method
    """

    pass


def augmented_largrange_multiplier(D, lmbda, tol=1e-7, maxiter=25000, verbose=2, inexact=True):
    """
    Augmented Lagrange Multiplier
    """
    obj = []
    Y = np.sign(D)
    norm_two = svd(Y, 1)[1]
    norm_inf = np.abs(Y).max() / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = _fro(D)
    tol_primal = 1e-6 * dnorm
    total_svd = 0
    mu = .5/norm_two
    rho = 6

    sv = 5
    svp = sv

    n = Y.shape[0]

    for iter in range(maxiter):
        primal_converged = False
        sv = sv + np.round(n * 0.1)
        primal_iter = 0
        while not primal_converged:
            Eraw = D - A + (1/mu) * Y
            Eupdate = np.maximum(
                Eraw - lmbda/mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
            U, S, V = svd(D - Eupdate + (1 / mu) * Y, sv)
            svp = (S > 1/mu).sum()
            if svp < sv:
                sv = np.min([svp + 1, n])
            else:
                sv = np.min([svp + round(.05 * n), n])
            Aupdate = np.dot(
                np.dot(U[:, :svp], np.diag(S[:svp] - 1/mu)), V[:svp, :])
            if primal_iter % 10 == 0 and verbose >= 2:
                print _fro(A - Aupdate)
            if (_fro(A - Aupdate) < tol_primal and _fro(E - Eupdate) < tol_primal) or (inexact and primal_iter > 5):
                primal_converged = True
                if verbose >= 2:
                    print "Primal Converged at Iter %d"%(primal_iter)
            A = Aupdate
            E = Eupdate
            primal_iter = primal_iter + 1
            total_svd = total_svd + 1
        Z = D - A - E
        Y = Y + mu * Z
        mu = rho * mu
        if np.sqrt((Z**2).sum()) / dnorm < tol:
            if verbose >= 2:
                print "Converged at Iter %d" % (iter)
            break
        else:
            if verbose >= 2:
                obj.append(_monitor(A, E, D))
    if verbose >= 2:
        return A, E, obj
    else:
        return A, E


def alternating_direction_method_of_multipliers(D, lmbda, rho=1., maxiter=25000, verbose=2, tol=1e-2):
    def soft_thresh(X, sigma):
        return np.maximum(X - sigma, 0) - np.maximum(-X - sigma, 0);
    obj = []
    m, n = D.shape
    A = D;
    E = D - A;
    W = np.ones(D.shape)/rho;
    rhoupdate = rho;
    for k in range(maxiter):
        U, S, V = svd(D-E-W);
        Aupdate = np.dot(np.dot(U, np.diag(soft_thresh(S, 1/rho))), V);
        Eupdate = soft_thresh(D-Aupdate-W, lmbda/rho);
        Wupdate = W + (Aupdate + Eupdate - D);
        primal_resid = _fro(Aupdate + Eupdate - D)
        dual_resid = rho*_fro(Eupdate - E)
        # this is from the stanford slide
        if primal_resid > 10*dual_resid:
            rhoupdate = 2*rho;
            Wupdate = Wupdate/2;
        elif dual_resid > 10*primal_resid:
            rhoupdate = rho/2;
            Wupdate = 2*Wupdate;
        else:
            rhoupdate = rho
        A = Aupdate;
        E = Eupdate;
        W = Wupdate;
        rho = rhoupdate;
        if primal_resid <= tol and dual_resid <= tol:
            if verbose >= 2:
                print 'Converged to tol=%e in %d iterations\n'%(tol, k)
            break;
        if verbose >= 2:
            obj.append(_monitor(A, E, D))
    if verbose >= 2:
        return A, E, obj
    else:
        return A, E

method = {"SVT": singular_value_thresholding,
          "ALM": augmented_largrange_multiplier,
          "ADMM": alternating_direction_method_of_multipliers,
          "APG": accelerate_proximal_gradient}


class RobustPCA(BaseEstimator, TransformerMixin):
    """
    Robust PCA
    """
    def __init__(self, alpha=.1, copy=True, method='svt'):
        """
        """
        pass

    def tranform():
        """
        Tranform
        """
        pass
