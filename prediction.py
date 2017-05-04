import numpy as np
import random
import os
import time
from glob import glob
import numpy as np
import scipy.misc
import argparse
import tensorflow as tf
import utils
from image import Images
from imagecache import ImageCache
from cyclegan import generator
CHECKPOINT_FILE = 'cyclegan.ckpt'
CHECKPOINT_DIR = './checkpoint/'

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-cd", "--check-dir", dest="checkpoint_dir", help="Directory where checkpoint file will be stored", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("-c", "--check", help="Name of the checkpoint file", type=str, default=CHECKPOINT_FILE)

    # Parse arguments
    args = parser.parse_args()

    # Raw print arguments
    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))

    return args


def get_model(sess):

    real_X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    real_Y = tf.placeholder(tf.float32, [None, 256, 256, 3])

    # genG(X) => Y            - fake_B
    genG = generator(real_X, name="generatorG")
    # genF(Y) => X            - fake_A
    genF = generator(real_Y, name="generatorF")

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)

    def predict():
        print('Inferring...')

        img = scipy.misc.imread('./sample.jpg', mode='RGB').astype(np.float32)
        img = (img / 127.5) - 1.

        resG = sess.run(genG, feed_dict={real_X: [img]})
        resF = sess.run(genF, feed_dict={real_Y: [img]})

        res = utils.inverse_transform(resG[0])
        scipy.misc.imsave('result.jpg', res)
        scipy.misc.imsave('result2.jpg', resF[0])

    return predict


def main():
    args = parseArguments()
    CHECKPOINT_FILE = args.check
    CHECKPOINT_DIR = args.checkpoint_dir

    sess = tf.Session()
    predict = get_model(sess)
    predict()


if __name__ == '__main__':
    main()
