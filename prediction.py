import cyclegan
import tensorflow as tf

def main():
  graph1 = tf.graph()
  with graph1.as_default():
    real_data = tf.placeholder(tf.float32, [None, 256, 256, 6], name='real_X_and_Y_images')


def define_model(graph, checkpoint_file):
# DEFINE OUR MODEL (generator only)
# -----------------------------------------------------
  with graph.as_default():
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

    # args = parseArguments()

    # # Raw print arguments
    # print("You are running the script with arguments: ")
    # for a in args.__dict__:
    #   print(str(a) + ": " + str(args.__dict__[a]))

    # A_TEST_DIR = args.testA
    # B_TEST_DIR = args.testB
    # CHECKPOINT_FILE = args.check
    # SAMPLES_DIR = args.samples

    ckpt = tf.train.get_checkpoint_state(checkpoint_file)

    # if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    #     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    # else:
    #     print("Created model with fresh parameters.")
    #     sess.run(tf.global_variables_initializer())

    # writer = tf.summary.FileWriter("./log", sess.graph)

    # dataX = glob(A_TEST_DIR + '/*.jpg')
    # dataY = glob(B_TEST_DIR + '/*.jpg')

    # np.random.shuffle(dataX)
    # np.random.shuffle(dataY)
    # batch_idxs = min(len(dataX), len(dataY))

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

