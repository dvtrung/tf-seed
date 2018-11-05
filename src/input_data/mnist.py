import gzip
import os
import shutil
import tempfile
import urllib

from src.input_data.base import BaseInputData
import tensorflow as tf
import numpy as np


class InputData(BaseInputData):
    DEFAULT_PARAMS = dict(
        data_path="data/mnist"
    )

    def __init__(self, params, mode, batch_size, dev=False):
        super().__init__(params, mode, batch_size, dev)

        self.images = tf.placeholder(tf.float32, [None, 784], name="images")
        self.labels = tf.placeholder(tf.int8, [None], name="labels")

        self._session = None

        if mode == "train" and not dev:
            images_file, labels_file = 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte'
        else:
            images_file, labels_file = 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'

        self._images_file = self.download(params.data_path, images_file)
        self._labels_file = self.download(params.data_path, labels_file)

        self.size = os.path.getsize(os.path.join(params.data_path, labels_file)) - 8

    def get_next(self):
        return self.images, self.labels

    def init_dataset(self):
        """Download and parse MNIST dataset."""

        def decode_image(image):
            # Normalize from [0, 255] to [0.0, 1.0]
            image = tf.decode_raw(image, tf.uint8)
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, [784])
            return image / 255.0

        def decode_label(label):
            label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
            label = tf.reshape(label, [])  # label is a scalar
            return tf.to_int32(label)

        images = tf.data.FixedLengthRecordDataset(
            self._images_file, 28 * 28, header_bytes=16).map(decode_image)
        labels = tf.data.FixedLengthRecordDataset(
            self._labels_file, 1, header_bytes=8).map(decode_label)

        dataset = tf.data.Dataset.zip((images, labels))
        self.batched_dataset = dataset.batch(self._params.batch_size)
        self._iterator = self.batched_dataset.make_initializable_iterator()

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        sess.run(self.iterator.initializer)

    def read32(self, bytestream):
        """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def download(self, directory, filename):
        """Download (and unzip) a file from the MNIST dataset if not already done."""
        filepath = os.path.join(directory, filename)
        if tf.gfile.Exists(filepath):
            return filepath
        if not tf.gfile.Exists(directory):
            tf.gfile.MakeDirs(directory)
        # CVDF mirror of http://yann.lecun.com/exdb/mnist/
        url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
        _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
        print('Downloading %s to %s' % (url, zipped_filepath))
        urllib.request.urlretrieve(url, zipped_filepath)
        with gzip.open(zipped_filepath, 'rb') as f_in, \
                tf.gfile.Open(filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(zipped_filepath)
        return filepath