## making a movie


import os
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from matplotlib import cm
import matplotlib
from robustpca import method
from plot_benchmark import mlabdefaults

if __name__ == "__main__":
    mlabdefaults()
    matplotlib.rcParams['savefig.dpi'] = 200
    files = []
    cache_path = '/tmp/robust_pca_tmp'
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    all_methods = method.keys()
    all_methods.append('MEAN')
    all_methods.append('PCA')
    for fname in all_methods:
        if not os.path.exists('%s/%s_tmp'%(cache_path, fname)):
            os.mkdir("%s/%s_tmp"%(cache_path, fname))
            mat = loadmat('./%s_background_subtraction.mat'%(fname))
            org = loadmat('./escalator_data.mat')['X'].reshape(
                160, 130, 200).swapaxes(0, 1)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(200):  # 50 frames
                ax.cla()
                ax.axis("off")
                ax.imshow(np.hstack((mat['A'][:, :, i],
                                     mat['E'][:, :, i], org[:, :, i])), cm.gray)
                fname_ = '%s/%s_tmp/_tmp%03d.png'%(cache_path, fname, i)
                print 'Saving frame', fname_
                fig.tight_layout()
                fig.savefig(fname_, bbox_inches="tight")
                files.append(fname_)
        print 'Making movie animation.mpg - this make take a while'
        os.system("mencoder 'mf://%s/%s_tmp/_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s_animation.mpg"%(cache_path, fname, fname))
