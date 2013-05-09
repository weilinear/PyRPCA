## making a movie

    
import os, sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

if __name__ == "__main__":
    files = []
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    if not os.path.exists('./tmp'):
        os.mkdir("tmp")
        mat = loadmat('./background_subtraction.mat')
        org = loadmat('./escalator_data.mat')['X'].reshape(160,130,200).swapaxes(0,1)
        for i in range(200):  # 50 frames
            ax.cla()
            ax.imshow(np.hstack((mat['A'][:,:,i], mat['E'][:,:,i], org[:,:,i])))
            fname = './tmp/_tmp%03d.png'%i
            print 'Saving frame', fname
            fig.savefig(fname)
            files.append(fname)
    print 'Making movie animation.mpg - this make take a while'
    os.system("mencoder 'mf://./tmp/_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg")
