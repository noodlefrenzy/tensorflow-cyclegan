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

# ensure reproducability
random_seed = 1234
np.random.seed(random_seed)

LOG_DIR = './log/'

BATCH_SIZE = 1
MAX_TRAIN_TIME_MINS = 600

SAMPLE_STEP = 10
SAMPLE_DIR = './samples/'
SAVE_STEP = 500
LOG_FREQUENCY = 1
TIME_CHECK_STEP = 100

L1_lambda = 10
MAX_STEPS = 100000

counter = 1
start_time = time.time()

SOFT_LABELS = False

PIPELINE_TWEAKS = {
    'random_flip': False,
    'random_contrast': False,
    'random_brightness': False,
    'random_saturation': False,
    'crop_size': 0
}

OPTIM_PARAMS = {
    'start_lr': [0.0002, 0.0002, 0.0002, 0.0002],
    'end_lr': [0., 0., 0., 0.],
    'lr_decay_start': [100000, 100000, 100000, 100000],
    'momentum': [0.5, 0.5, 0.5, 0.5]
}

CHECKPOINT_FILE = 'cyclegan.ckpt'
CHECKPOINT_DIR = './checkpoint/'

# READ INPUT PARAMS
def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Trains a CycleGAN using the given input sets and parameters.',
                                     epilog='''
                                     Arguments like "lrate" allow multiple values: 
                                     1 means "use the same for F/G/DX/DY",
                                     2 means "use the first for F/G and the second for DX/DY",
                                     3 means "use the first for F, the second for G, and the third for DX/DY",
                                     4 gives a distinct value to each (any higher than 4 are ignored)
                                     ''')

    # Optional arguments
    parser.add_argument('-log', '--logdir', dest='logdir', help='Log directory', default=LOG_DIR)
    parser.add_argument('-i', '--input', '--input_prefix', dest='input_prefix',
                                            help="Input prefix for tfrecords files (use to_tfrecords.py to generate).", required=True)
    parser.add_argument("-t", "--time", help="Max time (mins) to run training", type=int, default=60 * 10)
    parser.add_argument("-l", "--lrate", help="Starting learning rate (F, G, D_X, D_Y)", type=float, nargs="*", default=OPTIM_PARAMS['start_lr'], dest='start_lr')
    parser.add_argument("-m", "--momentum", help="Adam Momentum (F, G, D_X, D_Y)", type=float, nargs="*", default=OPTIM_PARAMS['momentum'])
    parser.add_argument("-lf", "--log-freq", dest="log_frequency", help= "How often writer should add summaries", default=LOG_FREQUENCY, type=int)
    parser.add_argument("-cd", "--check-dir", dest="checkpoint_dir", help="Directory where checkpoint file will be stored", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("-c", "--check", help="Name of the checkpoint file", type=str, default=CHECKPOINT_FILE)
    parser.add_argument('--end-lr', help='Ending learning rate (for decay) (F, G, D_X, D_Y)', type=float, nargs="*", default=OPTIM_PARAMS['end_lr'], dest='end_lr')
    parser.add_argument('--lr-decay-start', help='When (what step) to start decaying LR (F, G, D_X, D_Y)', type=int, nargs="*", default=OPTIM_PARAMS['lr_decay_start'], dest='lr_decay_start')
    parser.add_argument("-sl", "--softL", help="Set to True for random real labels around 1.0", action='store_true',
                                            default=SOFT_LABELS)
    parser.add_argument('-b', '--batch', '--batch-size', help='Batch size', type=int, default=BATCH_SIZE,
                                            dest='batch_size')
    parser.add_argument('-s', '--sample', '--sample-freq', dest='sample_freq', help='How often to write out sample images', type=int, default=SAMPLE_STEP)
    parser.add_argument('-sd', '--sample-dir', dest='sample_dir', help='Where write out sample images', type=str, default=SAMPLE_DIR)
    parser.add_argument('--checkpoint-freq', dest='checkpoint_freq', help='How often to save to the checkpoint file', type=int, default=SAVE_STEP)
    parser.add_argument('--ignore', '--ignore-checkpoint', dest='ignore_checkpoint', help='Ignore existing checkpoint file and start from scratch', action='store_true')
    parser.add_argument('--norm', help='Specify normalization type for non-Residual blocks', choices=['batch', 'instance', 'none'], default='batch')
    parser.add_argument('--rnorm', help='Specify normalization type for Residual blocks', choices=['batch', 'instance', 'none'], default='instance')
    parser.add_argument('--crop', help='Crop to (crop, crop) and then rescale images as they go into the pipeline', type=int, default=0)
    parser.add_argument('--random-flip', dest='random_flip', help='Whether to randomly flip images in the pipeline', action='store_true')
    parser.add_argument('--random-q', dest='random_q', help='Randomly mutate image qualitatively (saturation, contrast, brightness)', action='store_true')

    # Parse arguments
    args = parser.parse_args()

    # Raw print arguments
    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))
    
    print("Checkpoints will be saved in {}".format(os.path.join(args.checkpoint_dir, args.check)))
    return args

def to_image(data):
    return tf.image.convert_image_dtype((data + 1.) / 2., tf.uint8)


def batch_to_image(batch):
    return tf.map_fn(to_image, batch, dtype=tf.uint8)


# DEFINE OUR SAMPLING FUNCTIONS
# -------------------------------------------------------

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def sample_model(sess, idx, test_X, test_Y, testG, testF, testG_back, testF_back):
    # RUN THEM THROUGH THE MODEL
    x_val, y_val, y_samp, x_samp, y_cycle_samp, x_cycle_samp = sess.run(
        [test_X, test_Y, testG, testF, testG_back, testF_back])

    # GRAB THE RETURNED RESULTS AND COLOR CORRECT / MERGE DOWN TO SINGLE IMAGE FILE EACH
    pretty_x = merge(utils.inverse_transform(x_samp), [1, 1])
    pretty_y = merge(utils.inverse_transform(y_samp), [1, 1])
    pretty_x_cycle = merge(utils.inverse_transform(x_cycle_samp), [1, 1])
    pretty_y_cycle = merge(utils.inverse_transform(y_cycle_samp), [1, 1])

    if not os.path.isdir(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)

    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_X.jpg'.format(idx)), x_val[0])
    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_Y.jpg'.format(idx)), y_val[0])

    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_X_2Y.jpg'.format(idx)), y_samp[0])
    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_Y_2X.jpg'.format(idx)), x_samp[0])

    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_X_2Y_2X.jpg'.format(idx)), x_cycle_samp[0])
    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_Y_2X_2Y.jpg'.format(idx)), y_cycle_samp[0])


# DEFINE OUR CUSTOM LAYERS AND ACTIVATION FUNCTIONS
# -------------------------------------------------------
def batch_norm(x, name="batch_norm", reuse=False):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name,
                                        reuse=reuse)

def instance_norm(x, name='instance_norm', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        depth = x.get_shape()[3]
        scale = tf.get_variable('scale', [depth], initializer=tf.random_normal_initializer(1.0, 0.02))
        offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        inv = tf.rsqrt(variance + 1e-5)
        normalized = (x - mean) * inv
        return scale * normalized + offset

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


def do_norm(x, norm, name, reuse):
    if norm == 'batch':
        x = batch_norm(x, name + '_bn', reuse=reuse)
    elif norm == 'instance':
        x = instance_norm(x, name + '_in', reuse=reuse)
    return x

# DEFINE OUR GENERATOR
# -------------------------------------------------------
def generator(image, norm='batch', rnorm='instance', reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            tf.variable_scope(tf.get_variable_scope(), reuse=False)
            assert tf.get_variable_scope().reuse == False

        # CycleGAN code has two differences from typical Residual blocks
        # 1) They use instance normalization instead of batch normalization
        # 2) They are missing the final ReLU nonlinearity after joining with pass-through
        def residual_block(x, dim, ks=3, s=1, norm='instance', name='res', reuse=reuse):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1', reuse=reuse)
            y = do_norm(y, norm, name + '_1', reuse)
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2', reuse=reuse)
            y = do_norm(y, norm, name + '_2', reuse)
            return y + x

        s = 256
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c = tf.nn.relu(do_norm(conv2d(c, 32, 7, 1, padding='VALID', name='g_e1_c', reuse=reuse), norm, name + 'g_e1', reuse))
        c2 = tf.nn.relu(do_norm(conv2d(c, 64, 3, 2, name='g_e2_c', reuse=reuse), norm, 'g_e2', reuse))
        c3 = tf.nn.relu(do_norm(conv2d(c2, 128, 3, 2, name='g_e3_c', reuse=reuse), norm, 'g_e3', reuse))
        # define G network with 9 resnet blocks
        r1 = residual_block(c3, 128, norm=rnorm, name='g_r1', reuse=reuse)
        r2 = residual_block(r1, 128, norm=rnorm, name='g_r2', reuse=reuse)
        r3 = residual_block(r2, 128, norm=rnorm, name='g_r3', reuse=reuse)
        r4 = residual_block(r3, 128, norm=rnorm, name='g_r4', reuse=reuse)
        r5 = residual_block(r4, 128, norm=rnorm, name='g_r5', reuse=reuse)
        r6 = residual_block(r5, 128, norm=rnorm, name='g_r6', reuse=reuse)
        r7 = residual_block(r6, 128, norm=rnorm, name='g_r7', reuse=reuse)
        r8 = residual_block(r7, 128, norm=rnorm, name='g_r8', reuse=reuse)
        r9 = residual_block(r8, 128, norm=rnorm, name='g_r9', reuse=reuse)

        d1 = deconv2d(r9, 64, 3, 2, name='g_d1_dc', reuse=reuse)
        d1 = tf.nn.relu(do_norm(d1, norm, 'g_d1', reuse))

        d2 = deconv2d(d1, 32, 3, 2, name='g_d2_dc', reuse=reuse)
        d2 = tf.nn.relu(do_norm(d2, norm, 'g_d2', reuse))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c', reuse=reuse)
        pred = tf.nn.tanh(do_norm(pred, norm, 'g_pred', reuse))

        return pred


# DEFINE OUR DISCRIMINATOR
# -------------------------------------------------------
def discriminator(image, norm='batch', reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            tf.variable_scope(tf.get_variable_scope(), reuse=False)
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, 64, reuse=reuse, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(do_norm(conv2d(h0, 128, name='d_h1_conv', reuse=reuse), norm, 'd_1', reuse))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(do_norm(conv2d(h1, 256, name='d_h2_conv', reuse=reuse), norm, 'd_2', reuse))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(do_norm(conv2d(h2, 512, s=1, name='d_h3_conv', reuse=reuse), norm, 'd_3', reuse))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred', reuse=reuse)
        # h4 is (32 x 32 x 1)
        return h4

def save_model(saver, sess, counter):
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)
    saver.save(sess, path, global_step=counter)
    return path

def main():
    args = parseArguments()

    MAX_TRAIN_TIME_MINS = args.time
    CHECKPOINT_FILE = args.check
    CHECKPOINT_DIR = args.checkpoint_dir
    BATCH_SIZE = args.batch_size
    SAMPLE_STEP = args.sample_freq
    SAVE_STEP = args.checkpoint_freq
    SOFT_LABELS = args.softL
    LOG_DIR = args.logdir
    LOG_FREQUENCY = args.log_frequency
    PIPELINE_TWEAKS['random_flip'] = args.random_flip
    PIPELINE_TWEAKS['random_brightness'] = PIPELINE_TWEAKS['random_saturation'] = PIPELINE_TWEAKS[
        'random_contrast'] = args.random_q
    PIPELINE_TWEAKS['crop_size'] = args.crop

    if SOFT_LABELS:
        softL_c = 0.05
        #softL_c = np.random.normal(1,0.05)
        #if softL_c > 1.15: softL_c = 1.15
        #if softL_c < 0.85: softL_c = 0.85
    else:
        softL_c = 0.0
    print('Soft Labeling: ', softL_c)

    def parse_list(arglist, default):
        v1, v2, v3, v4 = default
        if len(arglist) == 1:
            v1 = v2 = v3 = v4 = arglist[0]
        elif len(arglist) == 2:
            v1 = v2 = arglist[0]
            v3 = v4 = arglist[1]
        elif len(arglist) == 3:
            v1 = arglist[0]
            v2 = arglist[1]
            v3 = v4 = arglist[2]
        elif len(arglist) > 3:
            v1, v2, v3, v4 = arglist[:4]
        return v1, v2, v3, v4
    OPTIM_PARAMS['start_lr'] = parse_list(args.start_lr, OPTIM_PARAMS['start_lr'])
    OPTIM_PARAMS['end_lr'] = parse_list(args.end_lr, OPTIM_PARAMS['end_lr'])
    OPTIM_PARAMS['momentum'] = parse_list(args.momentum, OPTIM_PARAMS['momentum'])
    OPTIM_PARAMS['lr_decay_start'] = parse_list(args.lr_decay_start, OPTIM_PARAMS['lr_decay_start'])
    sess = tf.Session()

    # DEFINE OUR MODEL AND LOSS FUNCTIONS
    # -------------------------------------------------------

    real_X = Images(args.input_prefix + '_trainA.tfrecords', batch_size=BATCH_SIZE, name='real_X').feed()
    real_Y = Images(args.input_prefix + '_trainB.tfrecords', batch_size=BATCH_SIZE, name='real_Y').feed()

    # genG(X) => Y            - fake_B
    genG = generator(real_X, norm=args.norm, rnorm=args.rnorm, name="generatorG")
    # genF(Y) => X            - fake_A
    genF = generator(real_Y, norm=args.norm, rnorm=args.rnorm, name="generatorF")
    # genF( genG(Y) ) => Y    - fake_A_
    genF_back = generator(genG, norm=args.norm, rnorm=args.rnorm, name="generatorF", reuse=True)
    # genF( genG(X)) => X     - fake_B_
    genG_back = generator(genF, norm=args.norm, rnorm=args.rnorm, name="generatorG", reuse=True)

    # DY_fake is the discriminator for Y that takes in genG(X)
    # DX_fake is the discriminator for X that takes in genF(Y)
    discY_fake = discriminator(genG, norm=args.norm, reuse=False, name="discY")
    discX_fake = discriminator(genF, norm=args.norm, reuse=False, name="discX")

    g_loss_G = tf.reduce_mean((discY_fake - tf.ones_like(discY_fake) * np.abs(np.random.normal(1.0,softL_c))) ** 2) \
            + L1_lambda * tf.reduce_mean(tf.abs(real_X - genF_back)) \
            + L1_lambda * tf.reduce_mean(tf.abs(real_Y - genG_back))

    g_loss_F = tf.reduce_mean((discX_fake - tf.ones_like(discX_fake) * np.abs(np.random.normal(1.0,softL_c))) ** 2) \
            + L1_lambda * tf.reduce_mean(tf.abs(real_X - genF_back)) \
            + L1_lambda * tf.reduce_mean(tf.abs(real_Y - genG_back))

    fake_X_sample = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_X_sample")
    fake_Y_sample = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_Y_sample")

    # DY is the discriminator for Y that takes in Y
    # DX is the discriminator for X that takes in X
    DY = discriminator(real_Y, norm=args.norm, reuse=True, name="discY")
    DX = discriminator(real_X, norm=args.norm, reuse=True, name="discX")
    DY_fake_sample = discriminator(fake_Y_sample, norm=args.norm, reuse=True, name="discY")
    DX_fake_sample = discriminator(fake_X_sample, norm=args.norm, reuse=True, name="discX")

    DY_loss_real = tf.reduce_mean((DY - tf.ones_like(DY) * np.abs(np.random.normal(1.0,softL_c))) ** 2)
    DY_loss_fake = tf.reduce_mean((DY_fake_sample - tf.zeros_like(DY_fake_sample)) ** 2)
    DY_loss = (DY_loss_real + DY_loss_fake) / 2

    DX_loss_real = tf.reduce_mean((DX - tf.ones_like(DX) * np.abs(np.random.normal(1.0,softL_c))) ** 2)
    DX_loss_fake = tf.reduce_mean((DX_fake_sample - tf.zeros_like(DX_fake_sample)) ** 2)
    DX_loss = (DX_loss_real + DX_loss_fake) / 2

    test_X = Images(args.input_prefix + '_testA.tfrecords', shuffle=False, name='test_A').feed()
    test_Y = Images(args.input_prefix + '_testB.tfrecords', shuffle=False, name='test_B').feed()

    testG = generator(test_X, norm=args.norm, rnorm=args.rnorm, name="generatorG", reuse=True)
    testF = generator(test_Y, norm=args.norm, rnorm=args.rnorm, name="generatorF", reuse=True)
    testF_back = generator(testG, norm=args.norm, rnorm=args.rnorm, name="generatorF", reuse=True)
    testG_back = generator(testF, norm=args.norm, rnorm=args.rnorm, name="generatorG", reuse=True)

    t_vars = tf.trainable_variables()
    DY_vars = [v for v in t_vars if 'discY' in v.name]
    DX_vars = [v for v in t_vars if 'discX' in v.name]
    g_vars_G = [v for v in t_vars if 'generatorG' in v.name]
    g_vars_F = [v for v in t_vars if 'generatorF' in v.name]

    # SETUP OUR SUMMARY VARIABLES FOR MONITORING
    # -------------------------------------------------------

    G_loss_sum = tf.summary.scalar("loss/G", g_loss_G)
    F_loss_sum = tf.summary.scalar("loss/F", g_loss_F)
    DY_loss_sum = tf.summary.scalar("loss/DY", DY_loss)
    DX_loss_sum = tf.summary.scalar("loss/DX", DX_loss)
    DY_loss_real_sum = tf.summary.scalar("loss/DY_real", DY_loss_real)
    DY_loss_fake_sum = tf.summary.scalar("loss/DY_fake", DY_loss_fake)
    DX_loss_real_sum = tf.summary.scalar("loss/DX_real", DX_loss_real)
    DX_loss_fake_sum = tf.summary.scalar("loss/DX_fake", DX_loss_fake)

    imgX = tf.summary.image('real_X', real_X, max_outputs=1)
    imgF = tf.summary.image('fake_X', genF, max_outputs=1)
    imgY = tf.summary.image('real_Y', real_Y, max_outputs=1)
    imgG = tf.summary.image('fake_Y', genG, max_outputs=1)

    # SETUP OUR TRAINING
    # -------------------------------------------------------

    def adam(loss, variables, start_lr, end_lr, lr_decay_start, start_beta, name_prefix):
        name = name_prefix + '_adam'
        global_step = tf.Variable(0, trainable=False)
        # The paper recommends learning at a fixed rate for several steps, and then linearly stepping down to 0
        learning_rate = (tf.where(tf.greater_equal(global_step, lr_decay_start),
                                tf.train.polynomial_decay(start_lr, global_step - lr_decay_start, lr_decay_start, end_lr,
                                                            power=1.0),
                                start_lr))
        lr_sum = tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

        learning_step = (tf.train.AdamOptimizer(learning_rate, beta1=start_beta, name=name).minimize(
            loss, global_step=global_step, var_list=variables))
        return learning_step, lr_sum

    print('Optimizing using {}'.format(OPTIM_PARAMS))
    DX_optim, DX_lr = adam(DX_loss, DX_vars,
                           OPTIM_PARAMS['start_lr'][2], OPTIM_PARAMS['end_lr'][2], OPTIM_PARAMS['lr_decay_start'][2], OPTIM_PARAMS['momentum'][2], 'D_X')

    DY_optim, DY_lr = adam(DY_loss, DY_vars,
                           OPTIM_PARAMS['start_lr'][3], OPTIM_PARAMS['end_lr'][3], OPTIM_PARAMS['lr_decay_start'][3], OPTIM_PARAMS['momentum'][3], 'D_Y')

    G_optim, G_lr = adam(g_loss_G, g_vars_G,
                         OPTIM_PARAMS['start_lr'][1], OPTIM_PARAMS['end_lr'][1], OPTIM_PARAMS['lr_decay_start'][1], OPTIM_PARAMS['momentum'][1], 'G')

    F_optim, F_lr = adam(g_loss_F, g_vars_F,
                         OPTIM_PARAMS['start_lr'][0], OPTIM_PARAMS['end_lr'][0], OPTIM_PARAMS['lr_decay_start'][0], OPTIM_PARAMS['momentum'][0], 'F')

    G_sum = tf.summary.merge(
        [G_loss_sum, G_lr]
    )
    F_sum = tf.summary.merge(
        [F_loss_sum, F_lr]
    )
    DY_sum = tf.summary.merge(
        [DY_loss_sum, DY_loss_real_sum, DY_loss_fake_sum, DY_lr]
    )
    DX_sum = tf.summary.merge(
        [DX_loss_sum, DX_loss_real_sum, DX_loss_fake_sum, DX_lr]
    )

    images_sum = tf.summary.merge([imgX, imgG, imgY, imgF])

    # CREATE AND RUN OUR TRAINING LOOP
    # -------------------------------------------------------

    print("Starting the time")
    timer = utils.Timer()

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./checkpoint/')

    if ckpt and ckpt.model_checkpoint_path and not args.ignore_checkpoint:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    cache_X = ImageCache(50)
    cache_Y = ImageCache(50)

    counter = 0
    try:
        while not coord.should_stop():

            # FORWARD PASS
            generated_X, generated_Y = sess.run([genF, genG])
            _, _, _, _, summary_str = sess.run([G_optim, DY_optim, F_optim, DX_optim, summary_op],
                    feed_dict={fake_Y_sample: cache_Y.fetch(generated_Y), fake_X_sample: cache_X.fetch(generated_X)})

            counter += 1
            print("[%4d] time: %4.4f" % (counter, time.time() - start_time))

            if np.mod(counter, LOG_FREQUENCY) == 0:
                print('writing')
                writer.add_summary(summary_str, counter)

            if np.mod(counter, SAMPLE_STEP) == 0:
                sample_model(sess, counter, test_X, test_Y, testG, testF, testG_back, testF_back)

            if np.mod(counter, SAVE_STEP) == 0:
                save_path = save_model(saver, sess, counter)
                print("Running for '{0:.2}' mins, saving to {1}".format(timer.elapsed() / 60, save_path))

            if np.mod(counter, SAVE_STEP) == 0:
                elapsed_min = timer.elapsed() / 60
                if (elapsed_min >= MAX_TRAIN_TIME_MINS):
                    print(
                        "Trained for '{0:.2}' mins and reached the max limit of {1}. Saving model.".format(elapsed_min,
                                                                                                        MAX_TRAIN_TIME_MINS))
                    coord.request_stop()

    except KeyboardInterrupt:
        print('Interrupted')
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        save_path = save_model(saver, sess, counter)
        print("Model saved in file: %s" % save_path)
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
