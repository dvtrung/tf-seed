import tensorflow as tf
from ..utils import model_utils, utils


class BaseModel(object):
    DEFAULT_PARAMS = {}

    def __call__(self,
                 hparams,
                 mode,
                 input_data,
                 **kwargs):
        self.hparams = hparams
        self.mode = mode
        self.train_mode = self.mode == tf.estimator.ModeKeys.TRAIN
        self.eval_mode = self.mode == tf.estimator.ModeKeys.EVAL
        self.infer_mode = self.mode == tf.estimator.ModeKeys.PREDICT

        self._input_data = input_data
        self._batch_size = self._assign_input()

        if self.train_mode:
            self._build_train_model(hparams, **kwargs)
        elif self.eval_mode:
            self._build_eval_model()
        elif self.infer_mode:
            self._build_infer_model()

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_train_model(self, params, **kwargs):
        self._loss = self._build_graph()

    def _build_eval_model(self):
        self._loss = self._build_graph()
        self._summary = tf.summary.merge([
            tf.summary.scalar('eval_loss', self.loss),
        ])

    def _build_infer_model(self):
        self._target_labels, self._target_seq_len = None, None
        self._build_graph()
        # self.summary = self._get_attention_summary()

    @property
    def iterator(self):
        return self._input.iterator

    def _assign_input(self):
        pass

    def _build_graph(self):
        return tf.constant(0)

    @classmethod
    def transfer_load(cls, sess, ckpt):
        """
        load with --transfer
        :param sess:
        :param ckpt:
        :return:
        """
        saver_variables = tf.global_variables()

        var_list = {var.op.name: var for var in saver_variables}
        del var_list['Variable_1']

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

    @classmethod
    def ignore_save_variables(cls):
        """
        list of variables that won't be saved
        :return:
        """
        return []

    @classmethod
    def trainable_variables(cls):
        """
        list of trainable variables
        :return:
        """
        return tf.trainable_variables()

    @property
    def ground_truth_label_placeholder(self):
        return [tf.no_op()]

    @property
    def predicted_label_placeholder(self):
        return [tf.no_op()]

    @property
    def ground_truth_label_len_placeholder(self):
        return [tf.constant(1)]

    @property
    def predicted_label_len_placeholder(self):
        return [tf.constant(1)]

    def get_extra_ops(self):
        return [tf.no_op()]