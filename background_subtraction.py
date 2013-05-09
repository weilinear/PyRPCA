import urllib2
from scipy.io import loadmat, savemat
from scipy.misc import imresize
import os
import numpy as np
from robustpca import *
### urllib

videoFname = "escalator_data.mat"
def download_video_clip():
    webFile = urllib2.urlopen(r'http://cvxr.com/tfocs/demos/rpca/escalator_data.mat')
    filename = './' + videoFname
    localFile = open(filename, 'w')
    localFile.write(webFile.read())
    localFile.close()

if __name__ == "__main__":
    if not os.path.exists('./' + videoFname):
        print 'Video file not found, downloading'
        download_video_clip()
    X = loadmat(videoFname)['X'].astype(np.double)
    nclip = X.shape[1]
    lmbda = .01

    A, E = augmented_largrange_multiplier(X, lmbda = lmbda , maxiter = 20, inexact = True)
    A = A.reshape(160, 130, X.shape[1]).swapaxes(0,1)
    E = E.reshape(160, 130, X.shape[1]).swapaxes(0,1)
    savemat("./background_subtraction.mat", {"A":A, "E":E})
    

    
    
        
