import tensorflow as tf
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
    dirToSave = "imgs/test_"
    """
    Just plots the output of the network, error, reconstructions, etc
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 6))
    ax[(0, 0)].imshow(real_A[0], interpolation='nearest')
    ax[(0, 1)].imshow(fake_B[0], interpolation='nearest')
    # ax[(0, 2)].imshow(create_image(np.squeeze(rec_A)), cmap=plt.cm.gray, interpolation='nearest')
    ax[(0, 0)].axis('off')
    ax[(0, 1)].axis('off')
    # ax[(0, 2)].axis('off')

    ax[(1, 0)].imshow(real_B[0], cmap=plt.cm.gray, interpolation='nearest')
    ax[(1, 1)].imshow(fake_A[0], cmap=plt.cm.gray, interpolation='nearest')
    # ax[(1, 2)].imshow(create_image(np.squeeze(rec_B)), cmap=plt.cm.gray, interpolation='nearest')
    ax[(1, 0)].axis('off')
    ax[(1, 1)].axis('off')
    # ax[(1, 2)].axis('off')
    fig.suptitle('Input | Fake ')
    if (isnotebook()):
        plt.show()
    else:
        # path = ''.join([dirToSave, "_1_", str(epoch), "_", str(iteration).zfill(4), '.png'])
        path = ''.join([dirToSave, "_", str(iteration).zfill(4), '.png'])
        print("Saving network output 1 to {0}".format(path))
        fig.savefig(path, dpi=100)
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10), linewidth=4)
        # D_A_plt, = plt.semilogy((self.D_A_loss_list), linewidth=4, ls='-', color='r', alpha=.5, label='D_A')
        # G_A_plt, = plt.semilogy((self.G_A_loss_list), linewidth=4, ls='-', color='b', alpha=.5, label='G_A')
        # D_B_plt, = plt.semilogy((self.D_B_loss_list), linewidth=4, ls='-', color='k', alpha=.5, label='D_B')
        # G_B_plt, = plt.semilogy((self.G_B_loss_list), linewidth=4, ls='-', color='g', alpha=.5, label='G_B')

        # axes = plt.gca()
        # leg = plt.legend(handles=[D_A_plt, G_A_plt, D_B_plt, G_B_plt], fontsize=20)
        # leg.get_frame().set_alpha(0.5)

        # if (isnotebook()):
        #    plt.show()
        # else:
        #    path = ''.join([dirToSave, "_2_", str(epoch), "_", str(iteration).zfill(4), '.png'])
        #    print("Saving network output 2 to {0}".format(path))
        #    fig.savefig(path, dpi=100)


def create_image(im):
    # scale the pixel values from [-1,1] to [0,1]
    return (im + 1) / 2
