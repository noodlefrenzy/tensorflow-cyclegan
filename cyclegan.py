import numpy as np
import random
import os
import time
from glob import glob
from six.moves import xrange
import numpy as np
import scipy.misc
import argparse
import tensorflow as tf
import utils

LOG_DIR = './log/'
A_DIR = './data/trainA/*.jpg'
B_DIR = './data/trainB/*.jpg'
A_TEST_DIR = './data/testA/*.jpg'
B_TEST_DIR = './data/testB/*.jpg'

MAX_TRAIN_TIME_MINS = 600

SAMPLE_STEP = 10
SAVE_STEP = 500
TIME_CHECK_STEP = 100

L1_lambda = 10
LEARNING_RATE = 0.0002
MOMENTUM = 0.5

counter = 1
start_time = time.time()
totalEpochs = 200

CHECKPOINT_FILE = './checkpoint/cyclegan.ckpt'

# READ INPUT PARAMS
def parseArguments():
	# Create argument parser
	parser = argparse.ArgumentParser()

	# Optional arguments
	parser.add_argument("-tA", "--trainA", help="Path to  trainA dir.", type=str, default=A_DIR)
	parser.add_argument("-tB", "--trainB", help="Path to  trainB dir.", type=str, default=B_DIR)
	parser.add_argument("-ttA", "--testA", help="Path to  testA dir.", type=str, default=A_TEST_DIR)
	parser.add_argument("-ttB", "--testB", help="Path to  testB dir.", type=str, default=B_TEST_DIR)
	parser.add_argument("-t", "--time", help="Max time (mins) to run training", type=int, default=60 * 10)
	parser.add_argument("-l", "--lrate", help="Learning rate", type=float, default=LEARNING_RATE)
	parser.add_argument("-c", "--check", help="Location of checkpoint  file where model will be stored", type=str, default=CHECKPOINT_FILE)

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

def prepare_sample_source():
	# WE WILL USE THE SAME SAMPLE EVERY TIME TO SEE EXPLICIT PROGRESS DURING TRAINING
	# WE GRAB THE FIRST IMAGE FROM THE TEST FOLDER AND LOAD IT
	sample_X_image_path = glob(A_TEST_DIR)[0]
	sample_Y_image_path = glob(B_TEST_DIR)[0]

	print("--------------------------------------------")
	print("sample_X_image_path=", sample_X_image_path)
	print("sample_Y_image_path=", sample_Y_image_path)
	print("--------------------------------------------")

	sample_X_image = scipy.misc.imread(sample_X_image_path, mode='RGB').astype(np.float)
	sample_Y_image = scipy.misc.imread(sample_Y_image_path, mode='RGB').astype(np.float)
	# RESIZE TO BE SURE
	sample_X_image = scipy.misc.imresize(sample_X_image, [256, 256])
	sample_Y_image = scipy.misc.imresize(sample_Y_image, [256, 256])
	sample_X_normalized = sample_X_image / 127.5 - 1.
	sample_Y_normalized = sample_Y_image / 127.5 - 1.
	sample_feed_data = np.concatenate( (sample_X_normalized, sample_Y_normalized), axis=2)
	sample_feed_data = [sample_feed_data]
	sample_feed_data = np.array(sample_feed_data).astype(np.float32)
	return sample_X_image, sample_Y_image, sample_feed_data    

def sample_model(epoch, idx, sample_X_image, sample_Y_image, sample_feed_data):

	# RUN THEM THROUGH THE MODEL
	generated_X_sample, generated_Y_sample = sess.run( 
		[ genF, genG ], 
		feed_dict={ real_data:sample_feed_data })

	# GRAB THE RETURNED RESULTS AND COLOR CORRECT / MERGE DOWN TO SINGLE IMAGE FILE EACH
	prettify_generated_X = merge( inverse_transform( generated_X_sample), [1,1])
	prettify_generated_Y = merge( inverse_transform( generated_Y_sample), [1,1])

	# CONVERT THE FIRST PASS OF GENERATOR RESULTS INTO DISCRETE IMAGES
	# TO PASS BACK THROUGH THE MODEL, COMPLETING THE ROUNDTRIP CYCLE TEST
	# I.E.  X=> G(X)=> F(G(X))=> X^   
	# and   Y=> F(Y)=> G(F(Y))=> Y^
	image_returnX2Y = merge(generated_X_sample, [1,1])
	image_returnY2X = merge(generated_Y_sample, [1,1])
	images_XY_return = np.concatenate((image_returnX2Y, image_returnY2X), axis=2)
	images_XY_return = [images_XY_return]
	images_XY_return = np.array(images_XY_return).astype(np.float32)

	# RUN THE RESULTS OF THE FIRST PASS THROUGH THE MODEL A SECOND TIME TO 
	# ATTEMPT TO CONVERT THEM BACK TO THEIR ORIGINAL MAPPINGS
	X_full_cycle, Y_full_cycle = sess.run(
		[genF, genG ], feed_dict={ real_data:images_XY_return}
	)
	
	# PREPARE THE FULLY CYCLED RESULTS FOR DISPLAY
	X_2Y_2X = merge( inverse_transform( X_full_cycle ), [1,1])
	Y_2X_2Y = merge( inverse_transform( Y_full_cycle ), [1,1])

	scipy.misc.imsave( './samples/X.jpg', sample_X_image )
	scipy.misc.imsave( './samples/Y.jpg', sample_Y_image )

	scipy.misc.imsave( './samples/X_2Y.jpg', prettify_generated_Y )
	scipy.misc.imsave( './samples/Y_2X.jpg', prettify_generated_X )

	scipy.misc.imsave( './samples/X_2Y_2X.jpg', X_2Y_2X )
	scipy.misc.imsave( './samples/Y_2X_2Y.jpg', Y_2X_2Y )


# DEFINE OUR CUSTOM LAYERS AND ACTIVATION FUNCTIONS
# -------------------------------------------------------
def batch_norm(x, name="batch_norm", reuse=False):
	return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name, reuse=reuse)


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


# DEFINE OUR RUNNING POOL OF 50 FAKES FOR THE DISCRIMINATOR
# -------------------------------------------------------
class ImagePool(object):
	def __init__(self, maxsize=50):
		self.maxsize = maxsize
		self.num_img = 0
		self.images = []

	def __call__(self, image):
		if self.maxsize == 0:
			return image
		if self.num_img < self.maxsize:
			self.images.append(image)
			return image
		if np.random.rand > 0.5:
			idx = int(np.random.rand * self.maxsize)
			tmp = copy.copy(self.images[idx])
			self.images[idx] = image
			return tmp
		else:
			return image


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
			y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1', reuse=reuse), name + '_bn1', reuse=reuse)
			y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
			y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2', reuse=reuse), name + '_bn2', reuse=reuse)
			return y + x

		s = 256
		# Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
		# The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
		# R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
		c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		c1 = tf.nn.relu(batch_norm(conv2d(c0, 32, 7, 1, padding='VALID', name='g_e1_c', reuse=reuse), 'g_e1_bn', reuse=reuse))
		c2 = tf.nn.relu(batch_norm(conv2d(c1, 64, 3, 2, name='g_e2_c', reuse=reuse), 'g_e2_bn', reuse=reuse))
		c3 = tf.nn.relu(batch_norm(conv2d(c2, 128, 3, 2, name='g_e3_c', reuse=reuse), 'g_e3_bn', reuse=reuse))
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


args = parseArguments()

# Raw print arguments
print("You are running the script with arguments: ")
for a in args.__dict__:
	print(str(a) + ": " + str(args.__dict__[a]))

A_DIR = args.trainA
B_DIR = args.trainB
A_TEST_DIR = args.testA
B_TEST_DIR = args.testB
MAX_TRAIN_TIME_MINS = args.time
LEARNING_RATE = args.lrate
CHECKPOINT_FILE = args.check

# DEFINE OUR MODEL AND LOSS FUNCTIONS
# -------------------------------------------------------

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

# DY_fake is the discriminator for Y that takes in genG(X)
# DX_fake is the discriminator for X that takes in genF(Y)
discY_fake = discriminator(genG, reuse=False, name="discY")
discX_fake = discriminator(genF, reuse=False, name="discX")

g_loss_G = tf.reduce_mean((discY_fake - tf.ones_like(discY_fake)) ** 2) \
		   + L1_lambda * tf.reduce_mean(tf.abs(real_X - genF_back)) \
		   + L1_lambda * tf.reduce_mean(tf.abs(real_Y - genG_back))

g_loss_F = tf.reduce_mean((discX_fake - tf.ones_like(discX_fake)) ** 2) \
		   + L1_lambda * tf.reduce_mean(tf.abs(real_X - genF_back)) \
		   + L1_lambda * tf.reduce_mean(tf.abs(real_Y - genG_back))

fake_X_sample = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_X_sample")
fake_Y_sample = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_Y_sample")

# DY is the discriminator for Y that takes in Y
# DX is the discriminator for X that takes in XD
DY = discriminator(real_Y, reuse=True, name="discY")
DX = discriminator(real_X, reuse=True, name="discX")
DY_fake_sample = discriminator(fake_Y_sample, reuse=True, name="discY")
DX_fake_sample = discriminator(fake_X_sample, reuse=True, name="discX")

DY_loss_real = tf.reduce_mean((DY - tf.ones_like(DY)) ** 2)
DY_loss_fake = tf.reduce_mean((DY_fake_sample - tf.zeros_like(DY_fake_sample)) ** 2)
DY_loss = (DY_loss_real + DY_loss_fake) / 2

DX_loss_real = tf.reduce_mean((DX - tf.ones_like(DX)) ** 2)
DX_loss_fake = tf.reduce_mean((DX_fake_sample - tf.zeros_like(DX_fake_sample)) ** 2)
DX_loss = (DX_loss_real + DX_loss_fake) / 2

test_X = tf.placeholder(tf.float32, [None, 256, 256, 3], name='testX')
test_Y = tf.placeholder(tf.float32, [None, 256, 256, 3], name='testY')

testY = generator(test_X, name="generatorG", reuse=True)
testX = generator(test_Y, name="generatorF", reuse=True)

t_vars = tf.trainable_variables()
DY_vars = [v for v in t_vars if 'discY' in v.name]
DX_vars = [v for v in t_vars if 'discX' in v.name]
g_vars_G = [v for v in t_vars if 'generatorG' in v.name]
g_vars_F = [v for v in t_vars if 'generatorF' in v.name]

# SETUP OUR SUMMARY VARIABLES FOR MONITORING
# -------------------------------------------------------

G_sum = tf.summary.scalar("g_loss_G", g_loss_G)
F_sum = tf.summary.scalar("g_loss_F", g_loss_F)
DY_loss_sum = tf.summary.scalar("DY_loss", DY_loss)
DX_loss_sum = tf.summary.scalar("DX_loss", DX_loss)
DY_loss_real_sum = tf.summary.scalar("DY_loss_real", DY_loss_real)
DY_loss_fake_sum = tf.summary.scalar("DY_loss_fake", DY_loss_fake)
DX_loss_real_sum = tf.summary.scalar("DX_loss_real", DX_loss_real)
DX_loss_fake_sum = tf.summary.scalar("DX_loss_fake", DX_loss_fake)

imgX = tf.summary.image('real_X', tf.transpose(real_X, perm=[0, 2, 3, 1]), max_outputs=3)
imgG = tf.summary.image('genG', tf.transpose(genG, perm=[0, 2, 3, 1]), max_outputs=3)
imgY = tf.summary.image('real_Y', tf.transpose(real_Y, perm=[0, 2, 3, 1]), max_outputs=3)
imgF = tf.summary.image('genF', tf.transpose(genF, perm=[0, 2, 3, 1]), max_outputs=3)

DY_sum = tf.summary.merge(
	[DY_loss_sum, DY_loss_real_sum, DY_loss_fake_sum]
)
DX_sum = tf.summary.merge(
	[DX_loss_sum, DX_loss_real_sum, DX_loss_fake_sum]
)

images_sum = tf.summary.merge([imgX, imgG, imgY, imgF])

# SETUP OUR TRAINING
# -------------------------------------------------------

DX_optim = tf.train.AdamOptimizer(LEARNING_RATE, MOMENTUM) \
	.minimize(DX_loss, var_list=DX_vars)

DY_optim = tf.train.AdamOptimizer(LEARNING_RATE, MOMENTUM) \
	.minimize(DY_loss, var_list=DY_vars)

G_optim = tf.train.AdamOptimizer(LEARNING_RATE, MOMENTUM) \
	.minimize(g_loss_G, var_list=g_vars_G)

F_optim = tf.train.AdamOptimizer(LEARNING_RATE, MOMENTUM) \
	.minimize(g_loss_F, var_list=g_vars_F)

# CREATE AND RUN OUR TRAINING LOOP
# -------------------------------------------------------

saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)

print("Starting the time")
timer = utils.Timer()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state('./checkpoint/')

if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
else:
	print("Created model with fresh parameters.")
	sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("./log", sess.graph)

is_early_break = False

sample_X_image, sample_Y_image, sample_feed_data = prepare_sample_source()

for epoch in xrange(totalEpochs):

	dataX = glob(A_DIR)
	dataY = glob(B_DIR)

	np.random.shuffle(dataX)
	np.random.shuffle(dataY)
	batch_idxs = min(len(dataX), len(dataY))

	pool = ImagePool(50)

	if is_early_break:
		print("Finishing early...")
		break

	for idx in xrange(0, batch_idxs):
		batch_files = zip(dataX[idx:(idx + 1)], dataY[idx:(idx + 1)])
		batch_images = [load_data(f) for f in batch_files]
		batch_images = np.array(batch_images).astype(np.float32)

		# FORWARD PASS
		generated_X, generated_Y = sess.run([genF, genG],
											feed_dict={real_data: batch_images})
		[generated_X, generated_Y] = pool([generated_X, generated_Y])

		# UPDATE  G
		_, summary_str = sess.run([G_optim, G_sum], feed_dict={real_data: batch_images})
		writer.add_summary(summary_str, counter)

		# UPDATE DY
		_, summary_str = sess.run([DY_optim, DY_sum],
								  feed_dict={real_data: batch_images,
											 fake_Y_sample: generated_Y})
		writer.add_summary(summary_str, counter)

		# UPDATE F
		_, summary_str = sess.run([F_optim, F_sum], feed_dict={real_data: batch_images})
		writer.add_summary(summary_str, counter)

		# UPDATE DX
		_, summary_str = sess.run([DX_optim, DX_sum],
								  feed_dict={real_data: batch_images,
											 fake_X_sample: generated_X})
		writer.add_summary(summary_str, counter)

		counter += 1
		print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch_idxs, time.time() - start_time))

		if np.mod(counter, SAMPLE_STEP) == 0:
			try:
				sample_model(epoch, idx, sample_X_image, sample_Y_image, sample_feed_data )
			except Exception as e:
				print("Error happend:")
				print(e)

		if np.mod(counter, SAVE_STEP) == 0:
			print("Running for '{0:.2}' mins, saving to {1}".format(timer.elapsed()/60, CHECKPOINT_FILE))
			saver.save(sess, CHECKPOINT_FILE)

		if np.mod(counter, SAVE_STEP) == 0:
			elapsed_min = timer.elapsed()/60
			if (elapsed_min >= MAX_TRAIN_TIME_MINS):
				print("Trained for '{0:.2}' mins and reached the max limit of {1}. Saving model.".format(elapsed_min, MAX_TRAIN_TIME_MINS))
				saver.save(sess, CHECKPOINT_FILE)
				is_early_break = True
				break

