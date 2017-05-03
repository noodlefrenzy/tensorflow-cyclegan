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

# ensure reproducability
random_seed = 1234
np.random.seed(random_seed)

LOG_DIR = './log/'

BATCH_SIZE = 1
MAX_TRAIN_TIME_MINS = 600

SAMPLE_STEP = 10
SAVE_STEP = 500
LOG_FREQUENCY = 1
TIME_CHECK_STEP = 100

L1_lambda = 10
LEARNING_RATE = 0.0002
MOMENTUM = 0.5
MAX_STEPS = 100000

counter = 1
start_time = time.time()

SOFT_LABELS = False

CHECKPOINT_FILE = 'cyclegan.ckpt'
CHECKPOINT_DIR = './checkpoint/'

# READ INPUT PARAMS
def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument('-log', '--logdir', dest='logdir', help='Log directory', default=LOG_DIR)
    parser.add_argument('-i', '--input', '--input_prefix', dest='input_prefix',
                                            help="Input prefix for tfrecords files.", required=True)
    parser.add_argument("-t", "--time", help="Max time (mins) to run training", type=int, default=60 * 10)
    parser.add_argument("-l", "--lrate", help="Learning rate", type=float, default=LEARNING_RATE)
    parser.add_argument("-lf", "--log-freq", dest="log_frequency", help= "How often writer should add summaries", default=LOG_FREQUENCY, type=int)
    parser.add_argument("-cd", "--check-dir", dest="checkpoint_dir", help="Directory where checkpoint file will be stored", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("-c", "--check", help="Name of the checkpoint file", type=str, default=CHECKPOINT_FILE)
    parser.add_argument('--end-lr', help='Ending learning rate (for decay)', type=float, default=0.0, dest='end_lr')
    parser.add_argument('--lr-decay-start', help='When (what step) to start decaying LR', type=int, default=100000, dest='lr_decay_start')
    parser.add_argument("-sl", "--softL", help="Set to True for random real labels around 1.0", action='store_true',
                                            default=SOFT_LABELS)
    parser.add_argument('-b', '--batch', '--batch-size', help='Batch size', type=int, default=BATCH_SIZE,
                                            dest='batch_size')
    parser.add_argument('-s', '--sample', '--sample-freq', dest='sample_freq', help='How often to write out sample images', type=int, default=SAMPLE_STEP)
    parser.add_argument('--checkpoint-freq', dest='checkpoint_freq', help='How often to save to the checkpoint file', type=int, default=SAVE_STEP)
    parser.add_argument('--ignore', '--ignore-checkpoint', dest='ignore_checkpoint', help='Ignore existing checkpoint file and start from scratch', action='store_true')

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


class Images():
    def __init__(self, tfrecords_file, image_size=256, batch_size=1, num_threads=2, shuffle=True, name=''):
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.shuffle = shuffle
        self.name = name

    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])
            reader = tf.TFRecordReader()

            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string),
                })

            image_buffer = features['image/encoded_image']
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = self.preprocess(image)
            if self.shuffle:
                images = tf.train.shuffle_batch(
                    [image], batch_size=self.batch_size, num_threads=self.num_threads,
                    capacity=100 + 3 * self.batch_size,
                    min_after_dequeue=100
                )
            else:
                images = tf.train.batch(
                    [image], batch_size=self.batch_size, num_threads=self.num_threads,
                    capacity=100 + 3 * self.batch_size
                )

            tf.summary.image('_input', images)
        return images

    def preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image / 127.5) - 1.
        image.set_shape([self.image_size, self.image_size, 3])
        return image


class ImageCache:
    def __init__(self, cache_size=30):
        self.cache_size = cache_size
        self.images = []

    def fetch(self, image):
        if self.cache_size == 0:
            return image

        p = random.random()
        if p > 0.5 and len(self.images) > 0:
            # use and replace old image.
            random_id = random.randrange(len(self.images))
            retval = self.images[random_id].copy()
            if len(self.images) < self.cache_size:
                self.images.append(image.copy())
            else:
                self.images[random_id] = image.copy()
            return retval
        else:
            if len(self.images) < self.cache_size:
                self.images.append(image.copy())
            return image


# DEFINE OUR LOAD DATA OPERATIONS
# -------------------------------------------------------
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


def sample_model(sess, idx):
    # RUN THEM THROUGH THE MODEL
    x_val, y_val, y_samp, x_samp, y_cycle_samp, x_cycle_samp = sess.run(
        [test_X, test_Y, testG, testF, testG_back, testF_back])

    # GRAB THE RETURNED RESULTS AND COLOR CORRECT / MERGE DOWN TO SINGLE IMAGE FILE EACH
    pretty_x = merge(inverse_transform(x_samp), [1, 1])
    pretty_y = merge(inverse_transform(y_samp), [1, 1])
    pretty_x_cycle = merge(inverse_transform(x_cycle_samp), [1, 1])
    pretty_y_cycle = merge(inverse_transform(y_cycle_samp), [1, 1])

    scipy.misc.imsave('./samples/{}_X.jpg'.format(idx), x_val[0])
    scipy.misc.imsave('./samples/{}_Y.jpg'.format(idx), y_val[0])

    scipy.misc.imsave('./samples/{}_X_2Y.jpg'.format(idx), y_samp[0])
    scipy.misc.imsave('./samples/{}_Y_2X.jpg'.format(idx), x_samp[0])

    scipy.misc.imsave('./samples/{}_X_2Y_2X.jpg'.format(idx), x_cycle_samp[0])
    scipy.misc.imsave('./samples/{}_Y_2X_2Y.jpg'.format(idx), y_cycle_samp[0])


# DEFINE OUR CUSTOM LAYERS AND ACTIVATION FUNCTIONS
# -------------------------------------------------------
def batch_norm(x, name="batch_norm", reuse=False):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name,
                                        reuse=reuse)


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

        def residual_block(x, dim, ks=3, s=1, name='res', reuse=reuse):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1', reuse=reuse), name + '_bn1',
                           reuse=reuse)
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2', reuse=reuse), name + '_bn2',
                           reuse=reuse)
            return y + x

        s = 256
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(
            batch_norm(conv2d(c0, 32, 7, 1, padding='VALID', name='g_e1_c', reuse=reuse), 'g_e1_bn', reuse=reuse))
        c2 = tf.nn.relu(batch_norm(conv2d(c1, 64, 3, 2, name='g_e2_c', reuse=reuse), 'g_e2_bn', reuse=reuse))
        c3 = tf.nn.relu(batch_norm(conv2d(c2, 128, 3, 2, name='g_e3_c', reuse=reuse), 'g_e3_bn', reuse=reuse))
        # define G network with 9 resnet blocks
        r1 = residual_block(c3, 128, name='g_r1', reuse=reuse)
        r2 = residual_block(r1, 128, name='g_r2', reuse=reuse)
        r3 = residual_block(r2, 128, name='g_r3', reuse=reuse)
        r4 = residual_block(r3, 128, name='g_r4', reuse=reuse)
        r5 = residual_block(r4, 128, name='g_r5', reuse=reuse)
        r6 = residual_block(r5, 128, name='g_r6', reuse=reuse)
        r7 = residual_block(r6, 128, name='g_r7', reuse=reuse)
        r8 = residual_block(r7, 128, name='g_r8', reuse=reuse)
        r9 = residual_block(r8, 128, name='g_r9', reuse=reuse)

        d1 = deconv2d(r9, 64, 3, 2, name='g_d1_dc', reuse=reuse)
        d1 = tf.nn.relu(batch_norm(d1, 'g_d1_bn', reuse=reuse))

        d2 = deconv2d(d1, 32, 3, 2, name='g_d2_dc', reuse=reuse)
        d2 = tf.nn.relu(batch_norm(d2, 'g_d2_bn', reuse=reuse))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c', reuse=reuse)
        pred = tf.nn.tanh(batch_norm(pred, 'g_pred_bn', reuse=reuse))

        return pred


# DEFINE OUR DISCRIMINATOR
# -------------------------------------------------------
def discriminator(image, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            tf.variable_scope(tf.get_variable_scope(), reuse=False)
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, 64, reuse=reuse, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(batch_norm(conv2d(h0, 128, name='d_h1_conv', reuse=reuse), 'd_bn1', reuse=reuse))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(batch_norm(conv2d(h1, 256, name='d_h2_conv', reuse=reuse), 'd_bn2', reuse=reuse))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(batch_norm(conv2d(h2, 512, s=1, name='d_h3_conv', reuse=reuse), 'd_bn3', reuse=reuse))
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

args = parseArguments()

MAX_TRAIN_TIME_MINS = args.time
LEARNING_RATE = args.lrate
CHECKPOINT_FILE = args.check
CHECKPOINT_DIR = args.checkpoint_dir
BATCH_SIZE = args.batch_size
SAMPLE_STEP = args.sample_freq
SAVE_STEP = args.checkpoint_freq
SOFT_LABELS = args.softL
LOG_DIR = args.logdir
LOG_FREQUENCY = args.log_frequency

if SOFT_LABELS:
	softL_c = 0.05
	#softL_c = np.random.normal(1,0.05)
	#if softL_c > 1.15: softL_c = 1.15
	#if softL_c < 0.85: softL_c = 0.85
else:
	softL_c = 0.0
print('Soft Labeling: ', softL_c)


sess = tf.Session()

# DEFINE OUR MODEL AND LOSS FUNCTIONS
# -------------------------------------------------------

real_X = Images(args.input_prefix + '_trainA.tfrecords', batch_size=BATCH_SIZE, name='real_X').feed()
real_Y = Images(args.input_prefix + '_trainB.tfrecords', batch_size=BATCH_SIZE, name='real_Y').feed()

# genG(X) => Y            - fake_B
genG = generator(real_X, name="generatorG")
# genF(Y) => X            - fake_A
genF = generator(real_Y, name="generatorF")
# genF( genG(Y) ) => Y    - fake_A_
genF_back = generator(genG, name="generatorF", reuse=True)
# genF( genG(X)) => X     - fake_B_
genG_back = generator(genF, name="generatorG", reuse=True)

# DY_fake is the discriminator for Y that takes in genG(X)
# DX_fake is the discriminator for X that takes in genF(Y)
discY_fake = discriminator(genG, reuse=False, name="discY")
discX_fake = discriminator(genF, reuse=False, name="discX")

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
DY = discriminator(real_Y, reuse=True, name="discY")
DX = discriminator(real_X, reuse=True, name="discX")
DY_fake_sample = discriminator(fake_Y_sample, reuse=True, name="discY")
DX_fake_sample = discriminator(fake_X_sample, reuse=True, name="discX")

DY_loss_real = tf.reduce_mean((DY - tf.ones_like(DY) * np.abs(np.random.normal(1.0,softL_c))) ** 2)
DY_loss_fake = tf.reduce_mean((DY_fake_sample - tf.zeros_like(DY_fake_sample)) ** 2)
DY_loss = (DY_loss_real + DY_loss_fake) / 2

DX_loss_real = tf.reduce_mean((DX - tf.ones_like(DX) * np.abs(np.random.normal(1.0,softL_c))) ** 2)
DX_loss_fake = tf.reduce_mean((DX_fake_sample - tf.zeros_like(DX_fake_sample)) ** 2)
DX_loss = (DX_loss_real + DX_loss_fake) / 2

test_X = Images(args.input_prefix + '_testA.tfrecords', shuffle=False, name='test_A').feed()
test_Y = Images(args.input_prefix + '_testB.tfrecords', shuffle=False, name='test_B').feed()

testG = generator(test_X, name="generatorG", reuse=True)
testF = generator(test_Y, name="generatorF", reuse=True)
testF_back = generator(testG, name="generatorF", reuse=True)
testG_back = generator(testF, name="generatorG", reuse=True)

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

DX_optim, DX_lr = adam(DX_loss, DX_vars, LEARNING_RATE, args.end_lr, args.lr_decay_start, MOMENTUM, 'D_X')

DY_optim, DY_lr = adam(DY_loss, DY_vars, LEARNING_RATE, args.end_lr, args.lr_decay_start, MOMENTUM, 'D_Y')

G_optim, G_lr = adam(g_loss_G, g_vars_G, LEARNING_RATE, args.end_lr, args.lr_decay_start, MOMENTUM, 'G')

F_optim, F_lr = adam(g_loss_F, g_vars_F, LEARNING_RATE, args.end_lr, args.lr_decay_start, MOMENTUM, 'F')

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
            sample_model(sess, counter)

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