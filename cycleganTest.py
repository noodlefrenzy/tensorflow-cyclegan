import time
from glob import glob
from six.moves import xrange
import numpy as np
import scipy.misc
import argparse
import os

import tensorflow as tf

A_TEST_DIR = './data/testA'
B_TEST_DIR = './data/testB'

counter = 1
start_time = time.time()

CHECKPOINT_FILE = './checkpoint'
SAMPLES_DIR = './samples'

# READ INPUT PARAMS
def parseArguments():
	# Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-A", "--testA", help="Path to  testA dir.", type=str, default=A_TEST_DIR)
    parser.add_argument("-B", "--testB", help="Path to  testB dir.", type=str, default=B_TEST_DIR)
    parser.add_argument("-c", "--check", help="Location of checkpoint file to be used", type=str, default=CHECKPOINT_FILE)
    parser.add_argument("-s", "--samples", help="Location of samples dir. to be used", type=str, default=SAMPLES_DIR)

	# Parse arguments
    args = parser.parse_args()

    return args

# DEFINE OUR LOAD DATA OPERATIONS
# -------------------------------------------------------
def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A / 127.5 - 1.
    img_B = img_B / 127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB


def load_image(image_path):
    img_A = scipy.misc.imread(image_path[0], mode='RGB').astype(np.float)
    img_B = scipy.misc.imread(image_path[1], mode='RGB').astype(np.float)
    return img_A, img_B


def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_A = img_A[h1:h1 + fine_size, w1:w1 + fine_size]
        img_B = img_B[h1:h1 + fine_size, w1:w1 + fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B


# DEFINE OUR SAMPLING FUNCTIONS
# -------------------------------------------------------
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


# DEFINE OUR CUSTOM LAYERS AND ACTIVATION FUNCTIONS
# -------------------------------------------------------

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d", reuse=False):
    with tf.variable_scope(name):
        return tf.layers.conv2d(input_,
                                filters=output_dim, kernel_size=ks, strides=(s, s),
                                padding=padding, kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                bias_initializer=None, reuse=reuse)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d", reuse=False):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(input_,
                                          filters=output_dim, kernel_size=ks, strides=(s, s), padding='SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                          bias_initializer=None, reuse=reuse)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


# DEFINE OUR GENERATOR
# -------------------------------------------------------
def generator(image, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            tf.variable_scope(tf.get_variable_scope(), reuse=False)
            assert tf.get_variable_scope().reuse == False

        def residule_block(x, dim, ks=3, s=1, name='res', reuse=reuse):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1', reuse=reuse), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2', reuse=reuse), name + '_bn2')
            return y + x

        s = 256
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(batch_norm(conv2d(c0, 32, 7, 1, padding='VALID', name='g_e1_c', reuse=reuse), 'g_e1_bn'))
        c2 = tf.nn.relu(batch_norm(conv2d(c1, 64, 3, 2, name='g_e2_c', reuse=reuse), 'g_e2_bn'))
        c3 = tf.nn.relu(batch_norm(conv2d(c2, 128, 3, 2, name='g_e3_c', reuse=reuse), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, 128, name='g_r1', reuse=reuse)
        r2 = residule_block(r1, 128, name='g_r2', reuse=reuse)
        r3 = residule_block(r2, 128, name='g_r3', reuse=reuse)
        r4 = residule_block(r3, 128, name='g_r4', reuse=reuse)
        r5 = residule_block(r4, 128, name='g_r5', reuse=reuse)
        r6 = residule_block(r5, 128, name='g_r6', reuse=reuse)
        r7 = residule_block(r6, 128, name='g_r7', reuse=reuse)
        r8 = residule_block(r7, 128, name='g_r8', reuse=reuse)
        r9 = residule_block(r8, 128, name='g_r9', reuse=reuse)

        d1 = deconv2d(r9, 64, 3, 2, name='g_d1_dc', reuse=reuse)
        d1 = tf.nn.relu(batch_norm(d1, 'g_d1_bn'))

        d2 = deconv2d(d1, 32, 3, 2, name='g_d2_dc', reuse=reuse)
        d2 = tf.nn.relu(batch_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c', reuse=reuse)
        pred = tf.nn.tanh(batch_norm(pred, 'g_pred_bn'))

        return pred


# DEFINE OUR MODEL (generator only)
# -----------------------------------------------------

real_data = tf.placeholder(tf.float32, [None, 256, 256, 6], name='real_X_and_Y_images')
real_X = real_data[:, :, :, :3]
real_Y = real_data[:, :, :, 3:6]

# genG(X) => Y            - fake_B
genG = generator(real_X, name="generatorG")
# genF( genG(Y) ) => Y    - fake_A_
genF_back = generator(genG, name="generatorF")
# genF(Y) => X            - fake_A
genF = generator(real_Y, name="generatorF", reuse=True)
# genF( genG(X)) => X     - fake_B_
genG_back = generator(genF, name="generatorG", reuse=True)

fake_X_sample = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_X_sample")
fake_Y_sample = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_Y_sample")

test_X = tf.placeholder(tf.float32, [None, 256, 256, 3], name='testX')
test_Y = tf.placeholder(tf.float32, [None, 256, 256, 3], name='testY')

testY = generator(test_X, name="generatorG", reuse=True)
testX = generator(test_Y, name="generatorF", reuse=True)

t_vars = tf.trainable_variables()

g_vars_G = [v for v in t_vars if 'generatorG' in v.name]
g_vars_F = [v for v in t_vars if 'generatorF' in v.name]

# SETUP OUR SUMMARY VARIABLES FOR MONITORING
# -------------------------------------------------------

imgX = tf.summary.image('real_X', tf.transpose(real_X, perm=[0, 2, 3, 1]), max_outputs=3)
imgG = tf.summary.image('genG', tf.transpose(genG, perm=[0, 2, 3, 1]), max_outputs=3)
imgY = tf.summary.image('real_Y', tf.transpose(real_Y, perm=[0, 2, 3, 1]), max_outputs=3)
imgF = tf.summary.image('genF', tf.transpose(genF, perm=[0, 2, 3, 1]), max_outputs=3)

images_sum = tf.summary.merge([imgX, imgG, imgY, imgF])

# CREATE AND RUN OUR GENERATON LOOP
# -------------------------------------------------------

saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

args = parseArguments()

# Raw print arguments
print("You are running the script with arguments: ")
for a in args.__dict__:
	print(str(a) + ": " + str(args.__dict__[a]))

A_TEST_DIR = args.testA
B_TEST_DIR = args.testB
CHECKPOINT_FILE = args.check
SAMPLES_DIR = args.samples

ckpt = tf.train.get_checkpoint_state(CHECKPOINT_FILE)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
else:
    print("Created model with fresh parameters.")
    sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("./log", sess.graph)

dataX = glob(A_TEST_DIR + '/*.jpg')
dataY = glob(B_TEST_DIR + '/*.jpg')

np.random.shuffle(dataX)
np.random.shuffle(dataY)
batch_idxs = min(len(dataX), len(dataY))

for idx in xrange(0, batch_idxs):
    batch_files = zip(dataX[idx:(idx + 1)], dataY[idx:(idx + 1)])
    batch_images = [load_data(f) for f in batch_files]
    batch_images = np.array(batch_images).astype(np.float32)

    # FORWARD PASS
    generated_X, generated_Y = sess.run([genF, genG],
                                        feed_dict={real_data: batch_images})

    scipy.misc.imsave('{0}/gen_{1:04d}_Y.jpg'.format(SAMPLES_DIR, idx), merge(inverse_transform(generated_Y), [1, 1]))

    scipy.misc.imsave('{0}/gen_{1:04d}_X.jpg'.format(SAMPLES_DIR, idx), merge(inverse_transform(generated_X), [1, 1]))

    counter += 1
    print("Step: [%2d] time: %4.4f" % (idx, time.time() - start_time))
