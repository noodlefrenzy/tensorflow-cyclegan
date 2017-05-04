import io
import logging
import os
from shutil import copyfile
import tensorflow as tf
import scipy.misc

import argparse

import random

def exists(x):
    for d in ['', 'trainA', 'trainB', 'testA', 'testB']:
        path = os.path.join(x, d)
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(
                '"%s" must be a valid directory containing trainA, trainB, testA, and testB dirs.' % x)
    return x

parser = argparse.ArgumentParser(description='Convert training and testing images to tfrecords files.')
subp = parser.add_subparsers(title='subcommands', description='valid subcommands', dest='command')

common = argparse.ArgumentParser(add_help=False)
common.add_argument('-o', '--output-prefix', help='Prefix file/path for output tfrecords files', required=True,
                    dest='output_prefix')
common.add_argument('-n', '--num-test', dest='num_test', type=int, help='Number of examples to include in test sets.', default=-1)
common.add_argument('--min', '--min-size', dest='min_size', type=int, help='Minimum image size (smaller images on either dimension will be filtered', default=0)
common.add_argument('--seed', dest='seed', type=int, help='Random seed to ensure repeatable shuffling', default=1337)
common.add_argument('--scale', dest='scale', type=int, help='Scale the image to a square of (scale, scale) size', default=0)
common.add_argument('--crop', dest='crop', type=int, help='Crop the image to a square of (crop, crop) size', default=0)
common.add_argument('--verbose', dest='verbose', action='store_true', help='Print out image details as they are stored')

prepped = subp.add_parser('prepped', aliases=['p'], help='For files already split into trainA/trainB/testA/testB.', parents=[common])
prepped.add_argument('--files', type=exists, help='Location of training files', required=True)

raw = subp.add_parser('raw', help='Turns two directories of images into the trainA/trainB/testA/testB directories, as well as the tfrecords files.', parents=[common])
raw.add_argument('--d1', '--dir1', '--directory1', dest='dir_1', help='Directory containing "Set A" of images.', required=True)
raw.add_argument('--d2', '--dir2', '--directory2', dest='dir_2', help='Directory containing "Set B" of images.', required=True)
raw.add_argument('-s', '--split', dest='split_pct', type=int, default=70, help='Split percent for training set.')
raw.add_argument('--outdir', '--out-dir', dest='out_dir', help='Will split the data and store in "prepped" format into this directory.', required=True)

def reader(path, shuffle=True):
    files = []

    for img_file in os.scandir(path):
        if img_file.name.lower().endswith('.jpg', ) and img_file.is_file():
            files.append(img_file.path)

    if shuffle:
        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable.
        shuffled_index = list(range(len(files)))
        random.shuffle(files)

        files = [files[i] for i in shuffled_index]

    return files

def raw_writer(dir_1, dir_2, min_size, scale_size, crop_size, split, out_dir, output_prefix, num_test=-1, verbose=False):
    files_1 = reader(dir_1)
    n_train_1 = int(len(files_1) * float(split) / 100.)
    files_2 = reader(dir_2)
    n_train_2 = int(len(files_2) * float(split) / 100.)
    train_1, test_1 = files_1[:n_train_1], files_1[n_train_1:]
    train_2, test_2 = files_2[:n_train_2], files_2[n_train_2:]

    for sub, files in [('trainA', train_1), ('trainB', train_2), ('testA', test_1), ('testB', test_2)]:
        cur_dir = os.path.join(out_dir, sub)
        os.makedirs(cur_dir, exist_ok=True)
        for file in files:
            copyfile(file, os.path.join(cur_dir, os.path.basename(file)))

    prepped_writer(out_dir, min_size, scale_size, crop_size, output_prefix, num_test=num_test, verbose=verbose)

def prepped_writer(root_path, min_size, scale_size, crop_size, output_prefix, num_test=-1, verbose=False):
    as_bytes = lambda data: tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
    as_example = lambda fn, data: tf.train.Example(features=tf.train.Features(feature={
        'image/file_name': as_bytes(tf.compat.as_bytes(os.path.basename(fn))),
        'image/encoded_image': as_bytes((data))
    }))

    for sub in ['trainA', 'trainB', 'testA', 'testB']:
        indir = os.path.join(root_path, sub)
        outfile = os.path.abspath('{}_{}.tfrecords'.format(output_prefix, sub))
        files = reader(indir)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        record_writer = tf.python_io.TFRecordWriter(outfile)

        if num_test != -1 and sub.startswith('test'):
            print('Limiting {} to {} files'.format(sub, num_test))
            files = files[:num_test]

        for ix, file in enumerate(files):
            im = scipy.misc.imread(file)
            if min_size > 0 and (im.shape[0] < min_size or im.shape[1] < min_size):
                print('Skipping image {} of size {}'.format(file, im.shape))
                continue

            if scale_size > 0:
                im = scipy.misc.imresize(im, size=(scale_size, scale_size), interp='bicubic')

            image_buf = io.BytesIO()
            as_img = scipy.misc.toimage(im)
            if crop_size > 0:
                if crop_size > as_img.height or crop_size > as_img.width:
                    logging.warning('Skipping image {} - ({}, {}) smaller than crop size {}'.format(file, as_img.width, as_img.height, crop_size))
                    continue

                max_bot = as_img.height - crop_size
                max_left = as_img.width - crop_size
                bot = random.randint(0, max_bot)
                left = random.randint(0, max_left)
                as_img = as_img.crop(box=(left, bot, left + crop_size, bot + crop_size))

            if verbose:
                print('Saving image {} of size {}x{} to {}'.format(file, as_img.width, as_img.height, outfile))
                
            as_img.save(image_buf, format='JPEG')
            example = as_example(file, image_buf.getvalue())
            record_writer.write(example.SerializeToString())

            if ix % 500 == 0:
                print('{}: Processed {}/{}.'.format(sub, ix, len(files)))
        print('Done.')
        record_writer.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    random.seed(args.seed)  # Set seed for repeatable shuffling.

    if args.command == 'raw':
        raw_writer(args.dir_1, args.dir_2, args.min_size, args.scale, args.crop, args.split_pct, args.out_dir, args.output_prefix, args.num_test, args.verbose)
    elif args.command == 'prepped':
        prepped_writer(args.files, args.min_size, args.scale, args.crop, args.output_prefix, args.num_test, args.verbose)
    else:
        parser.print_help()
