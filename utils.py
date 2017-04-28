import os
import numpy as np
from IPython import get_ipython

import matplotlib.pyplot as plt


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole?
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython?
            return False
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def plot_network_output(real_A, fake_B, real_B, fake_A, iteration):
    dirToSave = "testResults/"
    if not os.path.exists(dirToSave):
        os.makedirs(dirToSave)
    filePathName = dirToSave + "test_"
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 6))
    ax[(0, 0)].imshow(create_image(np.squeeze(real_A[0])), interpolation='nearest')
    ax[(0, 1)].imshow(create_image(np.squeeze(fake_B[0])), interpolation='nearest')
    ax[(0, 0)].axis('off')
    ax[(0, 1)].axis('off')
    ax[(1, 0)].imshow(create_image(np.squeeze(real_B[0])), cmap=plt.cm.gray, interpolation='nearest')
    ax[(1, 1)].imshow(create_image(np.squeeze(fake_A[0])), cmap=plt.cm.gray, interpolation='nearest')

    ax[(1, 0)].axis('off')
    ax[(1, 1)].axis('off')
    fig.suptitle('Input | Generated ')
    if (isnotebook()):
        plt.show()
    else:
        path = ''.join([filePathName, "_", str(iteration).zfill(4), '.png'])
        print("Saving network output 1 to {0}".format(path))
        fig.savefig(path, dpi=100)

def create_image(im):
    # scale the pixel values from [-1,1] to [0,1]
    return (im + 1) / 2
