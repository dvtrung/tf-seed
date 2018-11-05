import math

import tensorflow as tf
from src.models.base import BaseModel

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


class Model(BaseModel):
    DEFAULT_PARAMS = dict(
        hidden1_units=128,
        hidden2_units=32,
        num_classes=10,
    )

    def _assign_input(self):
        self._images, self._labels = self._input_data.iterator.get_next()
        return tf.shape(self._images)[0]

    def _build_graph(self):
        # Hidden 1
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal([IMAGE_PIXELS, self.hparams.hidden1_units],
                                    stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                name='weights')
            biases = tf.Variable(tf.zeros([self.hparams.hidden1_units]),
                                 name='biases')
            hidden1 = tf.nn.relu(tf.matmul(self._images, weights) + biases)

        # Hidden 2
        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([self.hparams.hidden1_units, self.hparams.hidden2_units],
                                    stddev=1.0 / math.sqrt(float(self.hparams.hidden1_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([self.hparams.hidden2_units]),
                                 name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

        # Linear
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([self.hparams.hidden2_units, self.hparams.num_classes],
                                    stddev=1.0 / math.sqrt(float(self.hparams.hidden2_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([self.hparams.num_classes]),
                                 name='biases')
            logits = tf.matmul(hidden2, weights) + biases

        labels = tf.to_int64(self._labels)

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        self._predicted_labels = tf.argmax(logits, axis=-1)

    @property
    def ground_truth_label_placeholder(self): return [self._labels]

    @property
    def predicted_label_placeholder(self): return [self._predicted_labels]

    def get_decode_fns(self):
        return [
            lambda d: str(d)
        ]