import numpy as np
import random
import tensorflow as tf

LOG_DIR = './log2/'
A_DIR = './data/trainA/*'
B_DIR = './data/trainB/*'

BATCH_SIZE = 4

MAX_ITERATION = 1000
NUM_CRITIC_TRAIN = 4

NUM_THREADS = 2

LAMBDA = 10
LAMBDA_CYCLE = 10
LEARNING_RATE = 0.001
BETA_1 = 0.5
BETA_2 = 0.9

SUMMARY_PERIOD = 25

#=====================================================
# DEFINE OUR INPUT PIPELINE FOR THE A / B IMAGE GROUPS
#=====================================================

def input_pipeline(filenames, batch_size, num_epochs=None, image_size=142, crop_size=128):

    with tf.device('/cpu:0'):
        filenames = tf.train.match_filenames_once(filenames)
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        reader = tf.WholeFileReader()
        filename, value = reader.read(filename_queue)
        image = tf.image.decode_jpeg(value, channels=3)

        processed = tf.image.resize_images(
            image,
            [image_size, image_size],
            tf.image.ResizeMethod.BILINEAR )

        processed = tf.image.random_flip_left_right(processed)
        processed = tf.random_crop(processed, [crop_size, crop_size, 3] )
        # CHANGE TO 'CHW' DATA_FORMAT FOR FASTER GPU PROCESSING
        processed = tf.transpose(processed, [2, 0, 1])
        processed = (tf.cast(processed, tf.float32) - 128.0) / 128.0

        images = tf.train.batch(
            [processed],
            batch_size = batch_size,
            num_threads = NUM_THREADS,
            capacity=batch_size * 5)

    return images

a = input_pipeline(A_DIR, BATCH_SIZE, 282, 256)
b = input_pipeline(B_DIR, BATCH_SIZE, 282, 256)

#=====================================================
# DEFINE OUR GENERATOR
#
# NOTE: We need to define additional helper functions
#       to supplement tensorflow:
#       instance_normalization and ResidualBlocks
#=====================================================

def instance_normalization(outputs, train=True):
    batch, channels, rows, cols = outputs.get_shape().as_list()
    var_shape = [rows]
    mu, sigma_sq = tf.nn.moments(outputs, [2, 3], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (outputs - mu) / (sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def ResBlock128(outputs, name=None):
    with tf.variable_scope(name):
        # WE MAY REQUIRED REFLECT PADDING AS IN HERE: https://github.com/vanhuyz/CycleGAN-TensorFlow/blob/master/ops.py
        res1 = tf.nn.relu(
            tf.layers.conv2d(outputs, filters=128,kernel_size=3, padding='same', data_format='channels_first', name='rb-conv2d-1')
        )
        #res1 = instance_normalization(res1)
        #res1 = tf.nn.relu(res1)
        res2 = tf.layers.conv2d(res1, filters=128, kernel_size=3, padding='same', data_format='channels_first', name='rb-conv-2d-2')
        return outputs + res2

# batch training versus testing, resuse=true and is_training=true
# when double called all need to be fleshed out more...
# also, an initial padding layer may help quality improvements later
# adding a tf.pad() then doing a tf.trim() back down....
# outputs = tf.pad( source, [ [0,0], [1, 1], [1, 1], [0, 0] ], "REFLECT" )
def build_generator(source, reuse=False) :

    batch, channels, image_size, _ = source.get_shape().as_list()

    with tf.variable_scope('generator'):

        # c7s1-32
        outputs = tf.layers.conv2d(source, filters=32, kernel_size=7, strides=1, padding='same',
            data_format='channels_first', name='c7s1-32-prebatch' )
        outputs = tf.layers.batch_normalization(outputs, training=True, reuse=reuse, epsilon=1e-5,
            momentum=0.9, name="c7s1-32")
        outputs = tf.nn.relu(outputs)

        # d64
        outputs = tf.layers.conv2d(outputs, filters=64, kernel_size=3, strides=2, padding='same',
            data_format='channels_first', name='d64-prebatch' )
        outputs = tf.layers.batch_normalization(outputs, training=True, reuse=reuse, epsilon=1e-5,
            momentum=0.9, name="d64")
        outputs = tf.nn.relu(outputs)

        # d128
        outputs = tf.layers.conv2d(outputs, filters=128, kernel_size=3, strides=2, padding='same',
            data_format='channels_first', name='d128-prebatch' )
        outputs = tf.layers.batch_normalization(outputs, training=True, reuse=reuse, epsilon=1e-5,
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
            data_format='channels_first', name='u64-prebatch' )
                
        outputs = tf.layers.batch_normalization(outputs, training=True, reuse=reuse, epsilon=1e-5,
            momentum=0.9, name="u64")
        outputs = tf.nn.relu(outputs)

        # u32
        outputs = tf.layers.conv2d_transpose(outputs, filters=32, kernel_size=3, 
            strides=2, padding='same', 
            data_format='channels_first', name='u32-prebatch' )
        
        outputs = tf.layers.batch_normalization(outputs, training=True, reuse=reuse, epsilon=1e-5,
            momentum=0.9, name="u32")
        outputs = tf.nn.relu(outputs)

        # c7s1-3
        outputs = tf.layers.conv2d(outputs, filters=3, kernel_size=7, padding='same', 
            data_format='channels_first', name='c7s1-3' )
        outputs = tf.nn.tanh(outputs, name='final-tanh')

        return outputs



with tf.variable_scope('generator_A2B') as a_to_b_scope :
    b_generator = build_generator(a)

with tf.variable_scope('generator_B2A') as b_to_a_scope :
    a_generator = build_generator(b)

with tf.variable_scope('generator_B2A',reuse=True) :
    a_identity = build_generator(b_generator,True)

with tf.variable_scope('generator_A2B',reuse=True) :
    b_identity = build_generator(a_generator,True)


#=====================================================
# DEFINE OUR DISCRIMINATOR
#=====================================================

def lrelu(outputs, name="lr"):
    return tf.maximum(outputs, 0.2*outputs, name=name)


def build_discriminator(source, reuse=None) :
    _, channels, _, _ = source.get_shape().as_list()

    with tf.variable_scope('discriminator'):

        #c64
        outputs = tf.layers.conv2d(source, filters=64, kernel_size=4, strides=2, padding='same',
            data_format='channels_first', name='c64' )
        outputs = lrelu(outputs, 'c64-lr')
        
        #c128
        outputs = tf.layers.conv2d(outputs, filters=128, kernel_size=4, strides=2, padding='same',
            data_format='channels_first', name='c128-prebatch' )
        outputs = tf.layers.batch_normalization(outputs, training=True, reuse=reuse, epsilon=1e-5,
            momentum=0.9, name="c128")
        outputs = lrelu(outputs, 'c128-lr')
        
        #c256
        outputs = tf.layers.conv2d(outputs, filters=256, kernel_size=4, strides=2, padding='same',
            data_format='channels_first', name='c256-prebatch' )
        outputs = tf.layers.batch_normalization(outputs, training=True, reuse=reuse, epsilon=1e-5,
            momentum=0.9, name="c256")
        outputs = lrelu(outputs, 'c256-lr')
        
        #c512
        outputs = tf.layers.conv2d(outputs, filters=512, kernel_size=4, strides=2, padding='same',
            data_format='channels_first', name='c512-prebatch' )
        outputs = tf.layers.batch_normalization(outputs, training=True, reuse=reuse, epsilon=1e-5,
            momentum=0.9, name="c512")
        outputs = lrelu(outputs, 'c512-lr')

        #c512REPEAT
        outputs = tf.layers.conv2d(outputs, filters=512, kernel_size=4, strides=2, padding='same',
            data_format='channels_first', name='c512REPEAT-prebatch' )
        outputs = tf.layers.batch_normalization(outputs, training=True, reuse=reuse, epsilon=1e-5,
            momentum=0.9, name="c512REPEAT")
        outputs = lrelu(outputs, 'c512REPEAT-lr')

        # FINAL RESHAPE
        outputs = tf.layers.conv2d(outputs, filters=1, kernel_size=1, padding='same',
            data_format='channels_first', name='final_reshape' )

    return outputs

with tf.variable_scope('discriminator_a') as scope:
    alpha = tf.random_uniform(shape=[BATCH_SIZE,1,1,1], minval=0.,maxval=1.)
    a_hat = alpha * a + (1.0-alpha) * a_generator

    v_a_real = build_discriminator(a)

    scope.reuse_variables()
    v_a_gen  = build_discriminator(a_generator)
    v_a_hat  = build_discriminator(a_hat, reuse=True)

with tf.variable_scope('discriminator_b') as scope:
    alpha = tf.random_uniform(shape=[BATCH_SIZE,1,1,1], minval=0.,maxval=1.)
    b_hat = alpha * b + (1.0-alpha) * b_generator

    v_b_real = build_discriminator(b)
    scope.reuse_variables()
    v_b_gen  = build_discriminator(b_generator)
    v_b_hat  = build_discriminator(b_hat, reuse=True)

disc_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator_')]
gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator_')]

#=====================================================
# DEFINE OUR LOSS FUNCTION
#=====================================================

d_optimizer = tf.train.AdamOptimizer(LEARNING_RATE,BETA_1,BETA_2)
g_optimizer = tf.train.AdamOptimizer(LEARNING_RATE,BETA_1,BETA_2)


W_a = tf.reduce_mean(v_a_real) - tf.reduce_mean(v_a_gen)
W_b = tf.reduce_mean(v_b_real) - tf.reduce_mean(v_b_gen)
W = W_a + W_b

GP_a = tf.reduce_mean(
        (tf.sqrt(tf.reduce_sum(tf.gradients(v_a_hat,a_hat)[0]**2,reduction_indices=[1,2,3]))-1.0)**2
     )
GP_b = tf.reduce_mean(
        (tf.sqrt(tf.reduce_sum(tf.gradients(v_b_hat,b_hat)[0]**2,reduction_indices=[1,2,3]))-1.0)**2
     )
GP = GP_a + GP_b

loss_c = -1.0*W + LAMBDA*GP
with tf.variable_scope('c_train') :
    gvs = d_optimizer.compute_gradients(loss_c,var_list=disc_vars)
    train_d_op = d_optimizer.apply_gradients(gvs)

loss_g_a = -1.0 * tf.reduce_mean(v_a_gen)
loss_g_b = -1.0 * tf.reduce_mean(v_b_gen)
loss_g = loss_g_a + loss_g_b

loss_cycle_a = tf.reduce_mean(
    tf.reduce_mean(tf.abs(a - a_identity),reduction_indices=[1,2,3])) # following the paper implementation.(divide by #pixels)
loss_cycle_b = tf.reduce_mean(
    tf.reduce_mean(tf.abs(b - b_identity),reduction_indices=[1,2,3])) # following the paper implementation.(divide by #pixels)
loss_cycle = loss_cycle_a + loss_cycle_b

with tf.variable_scope('g_train') :
    gvs = g_optimizer.compute_gradients(loss_g+LAMBDA_CYCLE*loss_cycle,var_list=gen_vars)
    train_g_op  = g_optimizer.apply_gradients(gvs)

#=====================================================
# SETUP TENSORBOARD
#=====================================================

tf.summary.image('real_a',tf.transpose(a,perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('fake_a',tf.transpose(a_generator, perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('identity_a',tf.transpose(a_identity,perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('real_b',tf.transpose(b,perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('fake_b',tf.transpose(b_generator, perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('identity_b',tf.transpose(b_identity,perm=[0,2,3,1]),max_outputs=10)

tf.summary.scalar('Estimated W',W)
tf.summary.scalar('gradient_penalty',GP)
tf.summary.scalar('loss_g', loss_g)
tf.summary.scalar('loss_cycle', loss_cycle)

# Summary Operations
summary_op = tf.summary.merge_all()

#=====================================================
# TRAIN OUR MODEL
#=====================================================

sess = tf.Session()
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

try:
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for step in range(MAX_ITERATION) :
        if coord.should_stop():
            break

        for _ in range(NUM_CRITIC_TRAIN):
            _ = sess.run(train_d_op)

        W_eval, GP_eval, loss_g_eval, loss_cycle_eval, _ = sess.run(
            [W,GP,loss_g,loss_cycle,train_g_op])

        print('%7d : W : %1.6f, GP : %1.6f, Loss G : %1.6f, Loss Cycle : %1.6f'%(
            step,W_eval,GP_eval,loss_g_eval,loss_cycle_eval))

        if( step % SUMMARY_PERIOD == 0 ) :
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

except Exception as e:
    coord.request_stop(e)
finally:
    coord.request_stop()
    coord.join(threads)
    sess.close()