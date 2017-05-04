import tensorflow as tf 

class Images():
    def __init__(self, tfrecords_file, image_size=256, batch_size=1, num_threads=2, shuffle=True, pipeline_tweaks=None, name=''):
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.shuffle = shuffle
        self.mods = pipeline_tweaks
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
        if self.mods and self.mods['random_flip']:
            image = tf.image.random_flip_left_right(image)
        if self.mods and self.mods['random_saturation']:
            image = tf.image.random_saturation(image, .95, 1.05)
        if self.mods and self.mods['random_brightness']:
            image = tf.image.random_brightness(image, .05)
        if self.mods and self.mods['random_contrast']:
            image = tf.image.random_contrast(image, .95, 1.05)

        if self.mods and self.mods['crop_size'] > 0:
            crop_size = self.mods['crop_size']
            wiggle = 8
            off_x, off_y = 25 - wiggle, 60 - wiggle
            crop_size_plus = crop_size + 2 * wiggle
            image = tf.image.resize_image_with_crop_or_pad(image, off_y + crop_size_plus, off_x + crop_size_plus)
            image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
            image = tf.random_crop(image, [crop_size, crop_size, 3])

        image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image / 127.5) - 1.
        image.set_shape([self.image_size, self.image_size, 3])
        return image
