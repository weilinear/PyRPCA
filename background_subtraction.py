import urllib2
from scipy.io import loadmat, savemat
from scipy.misc import imresize
import os
import numpy as np
from robustpca import *
from numpy.linalg import svd
### urllib

videoFname = "escalator_data.mat"


def download_video_clip():
    webFile = urllib2.urlopen(
        r'http://cvxr.com/tfocs/demos/rpca/escalator_data.mat')
    filename = './' + videoFname
    localFile = open(filename, 'w')
    localFile.write(webFile.read())
    localFile.close()

if __name__ == "__main__":
    if not os.path.exists('./' + videoFname):
        print 'Video file not found, downloading'
        download_video_clip()
    X = loadmat(videoFname)['X'].astype(np.double)/255.
    nclip = X.shape[1]
    # lmbda = .01
    lmbda = .01
    # lmbdas = {'APG':.01,
    #           'ALM':.01,
    #           'ADMM':.01,
    #           'SVT':.01}
    # save a basic value
    ################################### Mean ################################
    Xmean = X.mean(axis=1)
    A = np.tile(Xmean, (nclip, 1)).T
    E = X - A
    A = A.reshape(160, 130, X.shape[1]).swapaxes(0, 1) * 255.
    E = E.reshape(160, 130, X.shape[1]).swapaxes(0, 1) * 255.
    savemat("./MEAN_background_subtraction.mat", {"A": A, "E": E})
    ################################## SVD ##################################
    U, S, V = svd(X, full_matrices=False)
    r = 20
    A = np.dot(np.dot(U[:, :r], np.diag(S[:r])), V[:r, :])
    E = X - A
    A = A.reshape(160, 130, X.shape[1]).swapaxes(0, 1) * 255.
    E = E.reshape(160, 130, X.shape[1]).swapaxes(0, 1) * 255.
    savemat("./PCA_background_subtraction.mat", {"A": A, "E": E})
    ################################# Robust PCA ############################
    for fname in method.keys():
        m = method[fname]
        A, E = m(X, lmbda=lmbda, maxiter=100)
        A = A.reshape(160, 130, X.shape[1]).swapaxes(0, 1) * 255.
        E = E.reshape(160, 130, X.shape[1]).swapaxes(0, 1) * 255.
        savemat("./%s_background_subtraction.mat"%(fname), {"A": A, "E": E})
