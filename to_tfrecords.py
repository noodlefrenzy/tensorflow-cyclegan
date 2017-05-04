import logging
import os
from shutil import copyfile
import tensorflow as tf

import argparse

import random

random.seed(1337)  # Set seed for repeatable shuffling.

def exists(x):
    for d in ['', 'trainA', 'trainB', 'testA', 'testB']:
        path = os.path.join(x, d)
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(
                '"%s" must be a valid directory containing trainA, trainB, testA, and testB dirs.' % x)
    return x

parser = argparse.ArgumentParser(description='Convert training and testing images to tfrecords files.')
subp = parser.add_subparsers(title='subcommands', description='valid subcommands', dest='command')

prepped = subp.add_parser('prepped', aliases=['p'], help='For files already split into trainA/trainB/testA/testB.')
prepped.add_argument('--files', type=exists, help='Location of training files', required=True)
prepped.add_argument('-o', '--output-prefix', help='Prefix file/path for output tfrecords files', required=True,
                    dest='output_prefix')
prepped.add_argument('-n', '--num-test', dest='num_test', type=int, help='Number of examples to include in test sets.', default=-1)

raw = subp.add_parser('raw', help='Turns two directories of images into the trainA/trainB/testA/testB directories, as well as the tfrecords files.')
raw.add_argument('--d1', '--dir1', '--directory1', dest='dir_1', help='Directory containing "Set A" of images.', required=True)
raw.add_argument('--d2', '--dir2', '--directory2', dest='dir_2', help='Directory containing "Set B" of images.', required=True)
raw.add_argument('-s', '--split', dest='split_pct', type=int, default=70, help='Split percent for training set.')
raw.add_argument('--outdir', '--out-dir', dest='out_dir', help='Will split the data and store in "prepped" format into this directory.', required=True)
raw.add_argument('-o', '--output-prefix', help='Prefix file/path for output tfrecords files', required=True,
                    dest='output_prefix')
raw.add_argument('-n', '--num-test', dest='num_test', type=int, help='Number of examples to include in test sets.', default=-1)

def reader(path, shuffle=True):
    files = []

    for img_file in os.scandir(path):
        if img_file.name.endswith('.jpg') and img_file.is_file():
            files.append(img_file.path)

    if shuffle:
        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable.
        shuffled_index = list(range(len(files)))
        random.shuffle(files)

        files = [files[i] for i in shuffled_index]

    return files

def raw_writer(dir_1, dir_2, split, out_dir, output_prefix, num_test=-1):
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

    prepped_writer(out_dir, output_prefix, num_test=num_test)

def prepped_writer(root_path, output_prefix, num_test=-1):
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
            with tf.gfile.FastGFile(file, 'rb') as f:
                image_data = f.read()

            example = as_example(file, image_data)
            record_writer.write(example.SerializeToString())

            if ix % 500 == 0:
                print('{}: Processed {}/{}.'.format(sub, ix, len(files)))
        print('Done.')
        record_writer.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    if args.command == 'raw':
        raw_writer(args.dir_1, args.dir_2, args.split_pct, args.out_dir, args.output_prefix, args.num_test)
    elif args.command == 'prepped':
        prepped_writer(args.files, args.output_prefix, args.num_test)
    else:
        parser.print_help()
