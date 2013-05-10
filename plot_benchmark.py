from robustpca import *
import robustpca as rp
from time import time
from numpy.random import randn
from numpy.random import shuffle
import numpy as np
import matplotlib
import matplotlib.cm as cm
import cPickle as cP
k_colors = ["r", "b", "y", "m", "c", "g", "#FFA500", "k"];
k_markers = "o*dxs^vD";
from pylab import gcf
import pylab
import os

def mlabdefaults():
    matplotlib.rcParams['lines.linewidth'] = 1.5
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['font.size'] = 22.
    matplotlib.rcParams['font.family'] = "Times New Roman"
    matplotlib.rcParams['legend.fontsize'] = "small"
    matplotlib.rcParams['legend.fancybox'] = True
    matplotlib.rcParams['lines.markersize'] = 10
    matplotlib.rcParams['figure.figsize'] = 8, 5.6
    matplotlib.rcParams['legend.labelspacing'] = 0.1
    matplotlib.rcParams['legend.borderpad'] = 0.1
    matplotlib.rcParams['legend.borderaxespad'] = 0.2
    matplotlib.rcParams['font.monospace'] = "Courier New"


def synthesized_data(rank, dim, n_sample, sparse_ratio, sparse_mag):
    Basis = randn(rank, dim)
    W = randn(n_sample, rank)
    TrueMat = np.dot(W, Basis)
    # initalize a sparse matrix
    E = randn(TrueMat.size) * sparse_mag
    idx = np.arange(E.size)
    shuffle(idx)
    E[idx[int(sparse_ratio * E.size):]] = 0
    E = E.reshape(TrueMat.shape)
    # calculate the observation
    Observed = TrueMat + E  # 1000 x 100
    return Observed, TrueMat, E


def savefig(filename, fig=None):
    if fig is None:
        gcf().savefig(filename, bbox_inches='tight')
    else:
        fig.savefig(filename, bbox_inches='tight')


def generate_plot(x, results, xlabel=' ', ylabel=' ', keys=None, fname=None, me=1, title=None):
    mlabdefaults()
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    if keys is None:
        keys = results.keys()
    for i, m in enumerate(keys):
        y = results[m]
        ax.set_yscale('log')
        ax.plot(x[:len(y)], y,
                color=k_colors[i],
                linestyle="-",
                marker=k_markers[i], markevery=me)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_xlim(1,100)
    # ax.set_ylim(0,1.0)
    # ax.set_yticks(np.arange(0.0,1.1,0.1))
    ax.legend(keys, loc=4)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    if title is not None:
        fig.suptitle(title)
    if fname is None:
        pylab.show()
    else:
        savefig(fname, fig)

figurename = lambda x: "./benchfigures/%s"%(x)


def rank_experiment():
    for name in method.keys():
        result[name] = []
    x = [3, 5, 7, 9, 12]
    for rank in x:
        print "Rank %d"%(rank)
        mat, A, E = synthesized_data(rank=rank, dim=30,
                                     n_sample=1000, sparse_ratio=.05, sparse_mag=10)
        for name in method.keys():
            print "\t %s"%(name)
            time0 = time()
            m = method[name]
            A_, E_ = m(mat.T, lmbda=.1, verbose=0)
            escaped = time() - time0
            print "\t escaped %f"%(escaped)
            result[name].append(escaped)
    cP.dump({'x': x, 'results': result}, open(figurename(
        "rank_exp.pk"), 'w'), protocol=-1)
    generate_plot(x, results=result, xlabel='Rank', ylabel='Time',
                  keys=None, fname=figurename("rank_exp"))


def sparse_ratio_experiment():
    for name in method.keys():
        result[name] = []
    x = [0.01, 0.05, 0.1, 0.15, 0.2]
    for sr in x:
        print "sparse ratio %f"%(sr)
        mat, A, E = synthesized_data(rank=5, dim=30,
                                     n_sample=1000, sparse_ratio=sr, sparse_mag=10)
        for name in method.keys():
            print "\t %s"%(name)
            time0 = time()
            m = method[name]
            A_, E_ = m(mat.T, lmbda=.1, verbose=0)
            escaped = time() - time0
            print "\t escaped %f"%(escaped)
            result[name].append(escaped)
    cP.dump({'x': x, 'results': result}, open(figurename(
        "sparse_ratio_exp.pk"), 'w'), protocol=-1)
    generate_plot(x, results=result, xlabel='Sparsity Ratio',
                  ylabel='Time', keys=None, fname=figurename("sparse_ratio_exp"))


def sample_number_experiment():
    for name in method.keys():
        result[name] = []
    x = [300, 500, 800, 1000, 1500]
    for sr in x:
        print "number of samples %d"%(sr)
        mat, A, E = synthesized_data(rank=5, dim=30,
                                     n_sample=sr, sparse_ratio=0.05, sparse_mag=10)
        for name in method.keys():
            print "\t %s"%(name)
            time0 = time()
            m = method[name]
            A_, E_ = m(mat.T, lmbda=.1, verbose=0)
            escaped = time() - time0
            print "\t escaped %f"%(escaped)
            result[name].append(escaped)
    cP.dump({'x': x, 'results': result}, open(figurename(
        "data_number_exp.pk"), 'w'), protocol=-1)
    generate_plot(x, results=result, xlabel='Number of Training Samples',
                  ylabel='Time', keys=None, fname=figurename("data_number_exp"))


def convergency_experiment():
    result = {}
    for name in method.keys():
        result[name] = []
    maxiter = 100
    for sparse_ratio in [0.05, 0.1, 0.2]:
        for rank in [5, 10, 20]:
            cachename = figurename("iteration_exp_rank%d_sparse%0.2f.pk"%(rank, sparse_ratio));
            if not os.path.exists(cachename):
                mat, A, E = synthesized_data(rank=rank, dim=30,
                                             n_sample=1000, sparse_ratio=sparse_ratio, sparse_mag=10)
                for name in method.keys():
                    print "\t %s"%(name)
                    time0 = time()
                    m = method[name]
                    A_, E_, obj = m(
                        mat.T, lmbda=.1, verbose=2, maxiter=maxiter)
                    escaped = time() - time0
                    print "\t escaped %f"%(escaped)
                    result[name] = obj
                result['Optimal'] = np.ones(maxiter) * rp._monitor(A, E, A+E)
                cP.dump({'x': np.arange(maxiter), 'results': result}, open(cachename, 'w'), protocol=-1)
            else:
                tmp = cP.load(open(cachename, 'rb'))
                result = tmp['results']
            generate_plot(np.arange(maxiter), results=result, xlabel='Iteration', ylabel='Objective Value', keys=None, fname=figurename("iteration_exp_rank%d_sparse%0.2f.eps"%(rank, sparse_ratio)), me=3, title="Rank = %d, Sparse Error Ratio = %0.2f"%(rank, sparse_ratio))


if __name__ == "__main__":
    # rank_experiment()
    # sparse_ratio_experiment()
    # sample_number_experiment()
    convergency_experiment()
