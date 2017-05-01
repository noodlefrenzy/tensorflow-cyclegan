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

def create_traintest(inputdir,outputdir='./data',AB='A',train_frac = 0.75):
    '''
    inputs:
        inputdir - directory with images of one class (ex: sunny_beach or cloudy_beach)
        outputdir - directory to save the new train/test directories to. Default is cwd/data
        AB - for GANS, rename to either class 'A' or class 'B'
        train_frac - fraction of images in train vs test set

    outputs:
        New directories in current working directory of testA, testB, trainA, trainB
    '''
    #read in list of all files in inputdir and shuffle
    all_files = os.listdir(inputdir)
    np.random.shuffle(np.array(all_files))

    #seperate train/test lists
    train_size = np.int(len(all_files) * train_frac)
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]

    #create output directories and move image files to respective train/test dirs
    if not os.path.isdir(os.path.join(outputdir,'train'+AB)):
        os.mkdir(os.path.join(outputdir,'train'+AB))
    for file in train_files:
        os.rename(os.path.join(inputdir,file),os.path.join(outputdir,'train'+AB,file))
    if not os.path.isdir(os.path.join(outputdir,'test'+AB)):
        os.mkdir(os.path.join(outputdir,'test'+AB))
    for file in test_files:
        os.rename(os.path.join(inputdir,file),os.path.join(outputdir,'test'+AB,file))


