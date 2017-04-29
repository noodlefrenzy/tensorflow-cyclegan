import numpy as np
import random
import tensorflow as tf
import os
import sys
from utils import plot_network_output

LOG_DIR = './log/'

A_TEST_DIR = './data/testA/*.jpg'
B_TEST_DIR = './data/testB/*.jpg'

CHECKPT_FILE = './savedModel_inst_big.ckpt'

BATCH_SIZE = 1

MAX_ITERATION = 100000

NUM_THREADS = 1

SUMMARY_PERIOD = 10


# =====================================================
# DEFINE OUR INPUT PIPELINE FOR THE A / B IMAGE GROUPS
# =====================================================


def input_pipeline(filenames, batch_size, num_epochs=None, image_size=142, crop_size=256):
    with tf.device('/cpu:0'):
        filenames = tf.train.match_filenames_once(filenames)
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        reader = tf.WholeFileReader()
        filename, value = reader.read(filename_queue)

        image = tf.image.decode_jpeg(value, channels=3)

        processed = tf.image.resize_images(
            image,
            [image_size, image_size],
            tf.image.ResizeMethod.BILINEAR)

        processed = tf.image.random_flip_left_right(processed)
        processed = tf.random_crop(processed, [crop_size, crop_size, 3])
        # CHANGE TO 'CHW' DATA_FORMAT FOR FASTER GPU PROCESSING
        processed = tf.transpose(processed, [2, 0, 1])
        processed = (tf.cast(processed, tf.float32) - 128.0) / 128.0

        images = tf.train.batch(
            [processed],
            batch_size=batch_size,
            num_threads=NUM_THREADS,
            capacity=batch_size * 5)

    return images



a = input_pipeline(A_TEST_DIR, BATCH_SIZE, image_size=282, crop_size=256)
b = input_pipeline(B_TEST_DIR, BATCH_SIZE, image_size=282, crop_size=256)


# =====================================================
# DEFINE OUR GENERATOR
#
# NOTE: We need to define additional helper functions
#       to supplement tensorflow:
#       instance_normalization and ResidualBlocks
# =====================================================

def instance_normalization(outputs):
    batch, channels, rows, cols = outputs.get_shape().as_list()
    var_shape = [rows]
    mu, sigma_sq = tf.nn.moments(outputs, [2, 3], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (outputs - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift


def ResBlock128(outputs, name=None):
    with tf.variable_scope(name):
        # WE MAY REQUIRED REFLECT PADDING AS IN HERE: https://github.com/vanhuyz/CycleGAN-TensorFlow/blob/master/ops.py
        res1 = tf.layers.conv2d(outputs, filters=128, kernel_size=3, padding='same', data_format='channels_first',
                                name='rb-conv2d-1')

        res1 = instance_normalization(res1)
        res1 = tf.nn.relu(res1)
        res2 = tf.layers.conv2d(res1, filters=128, kernel_size=3, padding='same', data_format='channels_first',
                                name='rb-conv-2d-2')
        return outputs + res2


def build_generator(source, isTraining, reuse=False):
    batch, channels, image_size, _ = source.get_shape().as_list()

    with tf.variable_scope('generator'):
        # c7s1-32
        outputs = tf.layers.conv2d(source, filters=32, kernel_size=7, strides=1, padding='same',
                                   data_format='channels_first', name='c7s1-32-prebatch')
        outputs = tf.layers.batch_normalization(outputs, training=isTraining, reuse=reuse, epsilon=1e-5,
                                                momentum=0.9, name="c7s1-32")
        outputs = tf.nn.relu(outputs)

        # d64
        outputs = tf.layers.conv2d(outputs, filters=64, kernel_size=3, strides=2, padding='same',
                                   data_format='channels_first', name='d64-prebatch')
        outputs = tf.layers.batch_normalization(outputs, training=isTraining, reuse=reuse, epsilon=1e-5,
                                                momentum=0.9, name="d64")
        outputs = tf.nn.relu(outputs)

        # d128
        outputs = tf.layers.conv2d(outputs, filters=128, kernel_size=3, strides=2, padding='same',
                                   data_format='channels_first', name='d128-prebatch')
        outputs = tf.layers.batch_normalization(outputs, training=isTraining, reuse=reuse, epsilon=1e-5,
                                                momentum=0.9, name="d128")
        outputs = tf.nn.relu(outputs)

        # ADD RESIDUALBLOCKS (9 x R128)
        res1 = ResBlock128(outputs, 'res1')
        res2 = ResBlock128(res1, 'res2')
        res3 = ResBlock128(res2, 'res3')
        res4 = ResBlock128(res3, 'res4')
        res5 = ResBlock128(res4, 'res5')
        res6 = ResBlock128(res5, 'res6')
        res7 = ResBlock128(res6, 'res7')
        res8 = ResBlock128(res7, 'res8')
        res9 = ResBlock128(res8, 'res9')

        # u64
        outputs = tf.layers.conv2d_transpose(res9, filters=64, kernel_size=3,
                                             strides=2, padding='same',
                                             data_format='channels_first', name='u64-prebatch')

        outputs = tf.layers.batch_normalization(outputs, training=isTraining, reuse=reuse, epsilon=1e-5,
                                                momentum=0.9, name="u64")
        outputs = tf.nn.relu(outputs)

        # u32
        outputs = tf.layers.conv2d_transpose(outputs, filters=32, kernel_size=3,
                                             strides=2, padding='same',
                                             data_format='channels_first', name='u32-prebatch')

        outputs = tf.layers.batch_normalization(outputs, training=isTraining, reuse=reuse, epsilon=1e-5,
                                                momentum=0.9, name="u32")
        outputs = tf.nn.relu(outputs)

        # c7s1-3
        outputs = tf.layers.conv2d(outputs, filters=3, kernel_size=7, padding='same',
                                   data_format='channels_first', name='c7s1-3')
        outputs = tf.nn.tanh(outputs, name='final-tanh')

        return outputs


with tf.variable_scope('generator_A2B') as a_to_b_scope:
    b_generator = build_generator(a, False)

with tf.variable_scope('generator_B2A') as b_to_a_scope:
    a_generator = build_generator(b, False)

with tf.variable_scope('generator_B2A', reuse=True):
    a_identity = build_generator(b_generator, False, True)

with tf.variable_scope('generator_A2B', reuse=True):
    b_identity = build_generator(a_generator, False, True)



gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator_')]

# =====================================================
# SETUP TENSORBOARD
# =====================================================

tf.summary.image('real_a', tf.transpose(a, perm=[0, 2, 3, 1]), max_outputs=10)
tf.summary.image('fake_a', tf.transpose(a_generator, perm=[0, 2, 3, 1]), max_outputs=10)
tf.summary.image('identity_a', tf.transpose(a_identity, perm=[0, 2, 3, 1]), max_outputs=10)
tf.summary.image('real_b', tf.transpose(b, perm=[0, 2, 3, 1]), max_outputs=10)
tf.summary.image('fake_b', tf.transpose(b_generator, perm=[0, 2, 3, 1]), max_outputs=10)
tf.summary.image('identity_b', tf.transpose(b_identity, perm=[0, 2, 3, 1]), max_outputs=10)

# Summary Operations
summary_op = tf.summary.merge_all()

# Saver
saver = tf.train.Saver(max_to_keep=5)

sess = tf.Session()
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


try:

    summary_writer = tf.summary.FileWriter(LOG_DIR + "test/", sess.graph)
    saver.restore(sess, CHECKPT_FILE)

    threads = tf.train.start_queue_runners(sess=sess)
    for step in range(MAX_ITERATION):
        generatedA, generatedB, realA, realB = sess.run(
            [tf.transpose(a_generator, perm=[0, 2, 3, 1]), tf.transpose(b_generator, perm=[0, 2, 3, 1]),
             tf.transpose(a, perm=[0, 2, 3, 1]), tf.transpose(b, perm=[0, 2, 3, 1])])
        print('%7d :' % (
            step))
        plot_network_output(realA, generatedB, realB, generatedA, step)

        if (step % SUMMARY_PERIOD == 0):
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

except Exception as e:
   # coord.request_stop(e)
    print("Exception: " + e)
finally:
    sess.close()
