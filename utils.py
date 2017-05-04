import os
import numpy as np
from IPython import get_ipython
import time
from shutil import copyfile

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

class Timer():
  def __init__(self):
    self.start_time = time.time()

  def elapsed_reset(self):
    duration = self.elapsed()
    self.start_time = time.time()
    return duration

  def elapsed(self):
    end_time = time.time()
    duration = end_time - self.start_time
    return duration


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

def create_traintest(inputdir,outputdir='./data',AB='A',train_frac = 0.75, shuffle_seed=None):
    '''
    inputs:
        inputdir - directory with images of one class (ex: sunny_beach or cloudy_beach)
        outputdir - directory to save the new train/test directories to. Default is cwd/data
        AB - for GANS, rename to either class 'A' or class 'B'
        train_frac - fraction of images in train vs test set
        shuffle_seed - define random seed to make train/test split repeatable

    outputs:
        New directories in current working directory of testA, testB, trainA, trainB
    '''
    
    #read in list of all files in inputdir and shuffle
    if shuffle_seed:
        np.random.seed(shuffle_seed)
    all_files = os.listdir(inputdir)
    np.random.shuffle(np.array(all_files))

    #seperate train/test lists
    train_size = np.int(len(all_files) * train_frac)
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]

    #create output directories and move image files to respective train/test dirs
    os.makedirs(os.path.join(outputdir,'train'+AB),exist_ok=True)
    for file in train_files:
        copyfile(os.path.join(inputdir,file),os.path.join(outputdir,'train'+AB,file))
    os.makedirs(os.path.join(outputdir,'test'+AB),exist_ok=True)
    for file in test_files:
        copyfile(os.path.join(inputdir,file),os.path.join(outputdir,'test'+AB,file))


def to_image(data):
    return tf.image.convert_image_dtype((data + 1.) / 2., tf.uint8)

def batch_to_image(batch):
    return tf.map_fn(to_image, batch, dtype=tf.uint8)

def inverse_transform(images):
    return (images + 1.) / 2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img

def save_model(saver, sess, counter):
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)
    saver.save(sess, path, global_step=counter)
    return path

def inverse_transform(images):
    return (images + 1.) / 2.
