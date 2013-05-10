## making a movie

    
import os, sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from matplotlib import cm
from robustpca import method

if __name__ == "__main__":
    files = []
    cache_path = '/tmp/robust_pca_tmp'
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    for fname in method.keys():
        if not os.path.exists('%s/%s_tmp'%(cache_path, fname)):
            os.mkdir("%s/%s_tmp"%(cache_path, fname))
            mat = loadmat('./%s_background_subtraction.mat'%(fname))
            org = loadmat('./escalator_data.mat')['X'].reshape(160,130,200).swapaxes(0,1)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(200):  # 50 frames
                ax.cla()
                ax.axis("off")
                ax.imshow(np.hstack((mat['A'][:,:,i], mat['E'][:,:,i], org[:,:,i])), cm.gray)
                fname_ = '%s/%s_tmp/_tmp%03d.png'%(cache_path, fname,i)
                print 'Saving frame', fname_
                fig.savefig(fname_, bbox_inches="tight")
                files.append(fname_)
        print 'Making movie animation.mpg - this make take a while'
        os.system("mencoder 'mf://%s/%s_tmp/_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s_animation.mpg"%(cache_path, fname, fname))
