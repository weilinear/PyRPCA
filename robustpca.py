from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import scipy.sparse as sp

try:
    from pypropack import svdp
    raise ValueError
    svd = lambda X, k : svdp(X, k, 'L', kmax = max(100, 10 * k))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
except:
    from scipy.linalg import svd as svd_
    def svd(X, k = -1):
        U, S, V = svd_(X, full_matrices = False)
        if k < 0:
            return U,S,V
        else:
            return U[:,:k], S[:k], V[:k,:]

# The problem solved is
#                   min  : tau * (|A|_* + \lmbda |E|_1) + .5 * |(A,E)|_F^2
#              subject to: A + E = D
def _monitor(A, E, D, lmbda = 0.1):
    diags = svd(A, min(A.shape))[1]
    print "|A|_*" , np.abs(diags).sum()
    print "|A|_0" , (np.abs(diags) > 1e-6).sum()
    print "|E|_1" , np.abs(D - A).sum()
    print "|D-A-E|_F", _fro(D - A - E)
    return np.abs(diags).sum() + lmbda * np.abs(D - A).sum()
    
def _pos(A):
    return A * (A > 0)

def _fro(A):
    return np.sqrt((A * A).sum())

def singular_value_thresholding(D, maxiter = 25000, lmbda = 1.0, tau = 1e4, delta = .9, verbose = 2):
    """
    Singular Value Thresholding
    """
    # initialization
    _matshape = D.shape
    EPSILON_PRIMAL = 1e-5
    Y = np.zeros(shape = _matshape)
    A = np.zeros(shape = _matshape)
    E = np.zeros(shape = _matshape)
    rankA = 0
    obj = []
    for iter in range(maxiter):
        U, S, V = svd(Y, rankA+1)
        np.save("/tmp/Y.npy", Y)
        A = np.dot(np.dot(U, np.diag(_pos(S - tau))), V)
        E = np.sign(Y) * _pos(np.abs(Y) - lmbda * tau)
        M = D - A - E
        rankA = (S > tau).sum()
        Y = Y + delta * M
        if verbose >= 2:
            obj.append(_monitor(A, E, D))
        if _fro(D-A-E)/_fro(D) < EPSILON_PRIMAL:
            if verbose >= 2:
                print "Converged at iter %d"%iter
            break
    if verbose >= 2:
        return A, E, obj
    else:
        return A, E    

def accelerate_proximal_gradient(D, lmbda, maxiter = 25000, tol = 1e-7,
                                 lineSearchFlag = False,
                                 continuationFlag= True,
                                 eta = .9, mu = 1e-3, verbose = 2):
    """
    Accelerated Proximal Gradient (Partial SVD Version)
    """
    obj = []
    maxLineSearchIter = 200
    m, n = D.shape

    t_k = 1.
    t_km1 = 1.

    tau_0 = 2.

    X_km1_A = np.zeros(D.shape) 
    X_km1_E = np.zeros(D.shape) 
    X_k_A = np.zeros(D.shape) 
    X_k_E = np.zeros(D.shape) 

    if continuationFlag:
        if lineSearchFlag:
            mu_0 = eta * svd(D, 1)[1]
        else:
            mu_0 = svd(D, 1)[1]
        mu_k = mu_0 ;
        mu_bar = 1e-9 * mu_0   
    else:
        mu_k = mu ;

    tau_k = tau_0 ;

    converged = False ;
    iter = 0. ;

    sv = 5.;

    while not converged:

        Y_k_A = X_k_A + ((t_km1 - 1)/t_k)*(X_k_A-X_km1_A) ;
        Y_k_E = X_k_E + ((t_km1 - 1)/t_k)*(X_k_E-X_km1_E) ;

        if not lineSearchFlag:

            X_km1_A = Y_k_A - (1/tau_k)*(Y_k_A+Y_k_E-D) ;
            X_km1_E = Y_k_E - (1/tau_k)*(Y_k_A+Y_k_E-D) ;


            U ,S ,V = svd(X_km1_A);

            svp = (S > mu_k/tau_k).sum();
            if svp < sv:
                sv = min(svp + 1, n);
            else:
                sv = min(svp + round(0.05*n), n);

            X_kp1_A = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - mu_k/tau_k)), V[:svp, :])

            X_kp1_E = np.sign(X_km1_E) * _pos( np.abs(X_km1_E) - lmbda *mu_k / tau_k );

            rankA  = (S>mu_k/tau_k).sum();
            cardE = (np.abs(X_kp1_E)>0).sum();

        else:

            convergedLineSearch = 0 ;
            numLineSearchIter = 0 ;

            tau_hat = eta * tau_k ;

            while not convergedLineSearch:

                temp = np.dot(1/tau_hat, Y_k_A + Y_k_E - D)
                G_A = Y_k_A - temp
                G_E = Y_k_E - temp

                U, S, V = svd(G_A) ;

                SG_A = np.dot(np.dot(U, np.diag(_pos(S-mu_k/tau_hat))) , V);
                SG_E = np.sign(G_E) * _pos( np.abs(G_E) - lmbda*mu_k/tau_hat );

                F_SG = 0.5*norm(D-SG_A-SG_E,'fro')^2 ;
                Q_SG_Y = 0.5*tau_hat*((np.hstack((SG_A,SG_E))-np.hstack((G_A,G_E)))**2).sum() + (0.5-1/tau_hat)*((D-Y_k_A-Y_k_E)**2).sum()

                if F_SG <= Q_SG_Y:
                    tau_k = tau_hat ;
                    convergedLineSearch = True ;
                else:
                    tau_hat = min(tau_hat/eta,tau_0) ;

                numLineSearchIter = numLineSearchIter + 1 ;

                if (not convergedLineSearch) and numLineSearchIter >= maxLineSearchIter:
                    print 'Stuck in line search'
                    convergedLineSearch = True ;

            X_kp1_A = SG_A
            X_kp1_E = SG_E

            rankA  = (diagS>mu_k/tau_hat).sum()
            cardE = (abs(X_kp1_E)>0).sum()

        t_kp1 = 0.5*(1+np.sqrt(1+4*t_k*t_k)) ;

        X_km1_A = X_kp1_A + X_kp1_E - Y_k_A - Y_k_E ;
        Y_k_A = tau_k*(Y_k_A-X_kp1_A) + X_km1_A ;
        Y_k_E = tau_k*(Y_k_E-X_kp1_E) + X_km1_A ;

        s1 = np.sqrt((Y_k_A**2).sum()+(Y_k_E**2).sum())
        s2 = np.sqrt((X_kp1_A**2).sum()+(X_kp1_E**2).sum());
        stoppingCriterion = s1 / (tau_k*max(1,s2));
        if stoppingCriterion <= tol and iter > 10:
            converged = True ;

        if continuationFlag:
            mu_k = max(0.9*mu_k,mu_bar) ;

        t_km1 = t_k ;
        t_k = t_kp1 ;
        X_km1_A = X_k_A ;
        X_km1_E = X_k_E ;
        X_k_A = X_kp1_A ;
        X_k_E = X_kp1_E ;

        iter = iter + 1 ;

        if verbose >= 2:
            obj.append(_monitor(X_k_A, X_k_E, D))

        if (not converged) and iter >= maxiter:
            print 'Maximum iterations reached'
            converged = True ;
    if verbose >= 2:
        return X_k_A, X_k_E, obj
    else:
        return X_k_A, X_k_E

def dual_method():
    """
    Dual Method
    """
    
    pass

def augmented_largrange_multiplier(D, lmbda, tol = 1e-7, maxiter = 25000, verbose = 2, inexact = True):
    """
    Augmented Lagrange Multiplier
    """
    obj = []
    Y = np.sign(D)
    norm_two = svd(Y, 1)[1]
    norm_inf = np.abs(Y).max() / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A_hat = np.zeros(Y.shape)
    E_hat = np.zeros(Y.shape)
    # import pdb; pdb.set_trace()
    dnorm = _fro(D)
    tolProj = 1e-6 * dnorm
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
        while not primal_converged :
            temp_T = D - A_hat + (1/mu) * Y
            temp_E = np.maximum(temp_T - lmbda/mu, 0) + np.minimum(temp_T + lmbda / mu, 0)
            U, S, V = svd(D - temp_E + (1 / mu) * Y, sv)
            svp = (S > 1/mu).sum()
            if svp  < sv:
                sv = np.min([svp + 1, n])
            else:
                sv = np.min([svp + round(.05 * n), n])
            temp_A = np.dot(np.dot(U[:,:svp], np.diag(S[:svp] - 1/mu)), V[:svp,:])
            if primal_iter % 10 == 0 and verbose >= 2:
                print _fro(A_hat - temp_A)
            if (_fro(A_hat - temp_A) < tolProj and _fro(E_hat - temp_E) < tolProj) or (inexact and primal_iter > 5):
                primal_converged = True
                if verbose >= 2:
                    print "Primal Converged at Iter %d"%(primal_iter)
            A_hat = temp_A
            E_hat = temp_E
            primal_iter = primal_iter + 1
            total_svd = total_svd + 1
        Z = D - A_hat - E_hat
        Y = Y + mu * Z
        mu = rho * mu
        if np.sqrt((Z**2).sum()) / dnorm < tol:
            if verbose >= 2:
                print "Converged at Iter %d" %(iter)
            break
        else:
            if verbose >= 2:
                obj.append(_monitor(A_hat, E_hat, D))
                print "Iteration ", iter
                print "#SVD" , total_svd
                print "rank(A)", svp
                print "|E|_0", (E_hat > 1e-6).sum()
    if verbose >= 2:
        return A_hat, E_hat, obj
    else:
        return A_hat, E_hat

def soft_thresh(X, kappa):
    return np.maximum(X - kappa,0) - np.maximum(-X - kappa,0);

def alternating_direction_method_of_multipliers(D, lmbda, rho = 1., maxiter = 25000, verbose = 2, tol = 1e-1):
    obj = []
    m, n = D.shape

    # algorithm parameters and initial point

    # L = np.zeros(D.shape);
    L = D;
    S = D - L;
    W = np.ones(D.shape)/rho;

    rhonext = rho;

    for k in range(maxiter):

        # 1. prox of the nuclear norm
        U,Sig,V = svd(D-S-W);
        Lnext = np.dot(np.dot(U, np.diag(soft_thresh(Sig, 1/rho))), V);

        # 2. prox of the l1 norm
        Snext = soft_thresh(D-Lnext-W, lmbda/rho);

        # 3. residual running sum
        Wnext = W + (Lnext + Snext - D);

        # 4. evaluate primal and dual residuals
        # r^{k+1} = L^{k+1} + S^{k+1} - M
        # s^{k+1} = rho*(S^{k+1}-S^{k})
        primal_resid = _fro(Lnext + Snext - D)
        dual_resid = rho*_fro(Snext - S)

        # 5. automatically vary the penalty parameter
        if primal_resid > 10*dual_resid:
            rhonext = 2*rho;
            Wnext = Wnext/2;
        elif dual_resid > 10*primal_resid:
            rhonext = rho/2;
            Wnext = 2*Wnext;
        else:
            rhonext = rho

        # 6. update iterates
        L = Lnext;
        S = Snext;
        W = Wnext;
        rho = rhonext;

        # stopping criterion
        if primal_resid <= tol and dual_resid <= tol:
            if verbose >= 2:
                print 'Converged to tol=%e in %d iterations\n'%(tol, k)
            break;
        if verbose >= 2:
            obj.append(_monitor(L, S, D))

    if k == maxiter and verbose >= 2:
        print 'Failed to converge in maxiter=%d iterations\n'%maxiter
    if verbose >= 2:
        return L, S, obj
    else:
        return L, S

method = {"SVT" : singular_value_thresholding,
          "ALM" : augmented_largrange_multiplier,
          "ADMM" : alternating_direction_method_of_multipliers,
          "APG": accelerate_proximal_gradient}

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


