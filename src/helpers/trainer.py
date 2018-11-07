import tensorflow as tf
from ..utils import utils, ops_utils, model_utils


class TrainerHelper(object):
    def __init__(self, hparams, Model, InputData, mode):
        self.hparams = hparams
        self.mode = mode
        self.train_mode = self.mode == tf.estimator.ModeKeys.TRAIN
        self.eval_mode = self.mode == tf.estimator.ModeKeys.EVAL

        self.Model = Model
        self.InputData = InputData
        self.batch_size = self.hparams.eval_batch_size if self.eval_mode else self.hparams.batch_size
        self._batch_size = tf.Variable(
            self.batch_size,
            trainable=False,
            name="batch_size"
        )

        self.eval_batch_size = self.hparams.eval_batch_size
        self._eval_batch_size = tf.Variable(
            self.eval_batch_size,
            trainable=False,
            name="eval_batch_size"
        )

        if self.train_mode:  # init train set and dev set
            input_data_dev = InputData(
                self.hparams,
                tf.estimator.ModeKeys.EVAL,
                self._batch_size,
                dev=True
            )
            input_data_dev.init_dataset()

            input_data_train = InputData(
                self.hparams,
                tf.estimator.ModeKeys.TRAIN,
                self._eval_batch_size
            )
            input_data_train.init_dataset()
        else:
            input_data_train = None
            input_data_dev = None

        input_data_test = InputData(
            self.hparams,
            tf.estimator.ModeKeys.EVAL,
            self._eval_batch_size,
            dev=False
        )
        input_data_test.init_dataset()

        self._processed_inputs_count = tf.Variable(0, trainable=False)
        self.processed_inputs_count = 0
        self.increment_inputs_count = tf.assign(
            self._processed_inputs_count,
            self._processed_inputs_count + self.batch_size)

        self._global_step = tf.Variable(0, trainable=False)

        self._input_data_train = input_data_train
        self._input_data_test = input_data_test
        self._input_data_dev = input_data_dev

        self._eval_count = 0

        self._train_model = self._test_model = self._dev_model = None

    def init(self, sess):
        self.processed_inputs_count = sess.run(self._processed_inputs_count)
        self.reset_train_iterator(sess)
        self._input_data_test.reset_iterator(sess)
        if self._input_data_dev is not None:
            self._input_data_dev.reset_iterator(sess)

    def build_model(self, eval=False):
        if self.train_mode:
            with tf.variable_scope(tf.get_variable_scope()):
                self.learning_rate = tf.constant(self.hparams.learning_rate)
                # self.learning_rate = self._get_learning_rate_warmup(self.hparams)
                self.learning_rate = self._get_learning_rate_decay()

                # build model
                model = self.Model()
                model(
                    self.hparams,
                    tf.estimator.ModeKeys.TRAIN,
                    self._input_data_train
                )
                opt = utils.get_optimizer(self.hparams, self.learning_rate)
                self.loss = model.loss
                self.params = model.trainable_variables()

                # compute gradient & update variables
                gradients = opt.compute_gradients(
                    self.loss,
                    var_list=self.params,
                    colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops
                )

                clipped_grads, grad_norm_summary, grad_norm = model_utils.gradient_clip(
                    [grad for grad, _ in gradients], max_gradient_norm=self.hparams.max_gradient_norm
                )

                self.grad_norm = grad_norm

                self.update = opt.apply_gradients(zip(clipped_grads, self.params), self._global_step)

            self._summary = tf.summary.merge([
                tf.summary.scalar('train_loss', self.loss),
                tf.summary.scalar("learning_rate", self.learning_rate),
            ])

            self._train_model = model

            # init dev model
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self._dev_model = self.Model()
                self._dev_model(
                    self.hparams,
                    tf.estimator.ModeKeys.EVAL,
                    self._input_data_dev
                )

        # init test model
        if eval > 0 or self.eval_mode:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self._test_model = self.Model()
                self._test_model(
                    self.hparams,
                    tf.estimator.ModeKeys.EVAL,
                    self._input_data_test
                )
                self._eval_summary = tf.no_op()

        if self.hparams.verbose:
            self.print_logs()

    def reset_train_iterator(self, sess):
        if self.train_mode:
            self._input_data_train.reset_iterator(
                sess,
                skip=self.processed_inputs_count % self.data_size,
                # shuffle=self.epoch > 5
            )
            self._train_model._assign_input()

    def print_logs(self):
        print("# Variables")
        for param in tf.global_variables():
            print("  %s, %s, %s" % (
                param.name,
                str(param.get_shape()),
                param.op.device
            ))

        if self.train_mode:
            print("# Trainable variables")
            for param in self.params:
                print("  %s, %s, %s" % (
                    param.name,
                    str(param.get_shape()),
                    param.op.device
                ))

    def train(self, sess):
        model = self._train_model
        try:
            self.processed_inputs_count, loss, _, summary, _, ground_truth_labels, predicted_labels, extra_ops, _ = sess.run(
                [
                    self._processed_inputs_count,
                    self.loss,
                    self._global_step,
                    self._summary,
                    self.increment_inputs_count,
                    model.ground_truth_label_placeholder,
                    model.predicted_label_placeholder,
                    model.get_extra_ops(),
                    self.update
                ]
            )

        except tf.errors.OutOfRangeError:
            # end of epoch -> reset iterator
            self.processed_inputs_count, _ = \
                sess.run([self._processed_inputs_count, self.increment_inputs_count])
            self.reset_train_iterator(sess)
            return self.train(sess)

        return loss, summary

    def eval(self, sess, dev=False):
        model = self._dev_model if dev else self._test_model
        ground_truth_labels, predicted_labels, ground_truth_len, predicted_len, \
            loss, summary, extra_ops = \
            sess.run([
                model.ground_truth_label_placeholder,
                model.predicted_label_placeholder,
                model.ground_truth_label_len_placeholder,
                model.predicted_label_len_placeholder,
                model.loss,
                self._eval_summary,
                model.get_extra_ops()
            ])

        return ground_truth_labels, predicted_labels, ground_truth_len, predicted_len

    def eval_all(self, sess, dev=False):
        """
        Iterate through dataset and return final accuracy

        Returns
            dictionary of key: id, value: accuracy
        """
        lers = {}
        decode_fns = self._test_model.get_decode_fns()
        metrics = self.hparams.metrics.split(',')

        input_data = self._input_data_dev if dev else self._input_data_test

        if input_data is None:
            return None
        input_data.reset_iterator(sess)

        while True:
            try:
                ground_truth_labels, predicted_labels, ground_truth_len, predicted_len = self.eval(sess, dev)
                for acc_id, (gt_labels, p_labels, gt_len, p_len) in \
                        enumerate(zip(ground_truth_labels, predicted_labels, ground_truth_len, predicted_len)):
                    if acc_id not in lers:
                        lers[acc_id] = []
                    for i in range(len(gt_labels)):
                        ler, _, _ = ops_utils.evaluate(
                            gt_labels[i],  # [:gt_len[i]],
                            p_labels[i],  # [:p_len[i]],
                            decode_fns[acc_id],
                            metrics[acc_id]
                        )

                    if ler is not None:
                        lers[acc_id].append(ler)
            except tf.errors.OutOfRangeError:
                break

        return {acc_id: sum(lers[acc_id]) / len(lers[acc_id]) for acc_id in lers}

    def _get_learning_rate_warmup(self, hparams):
        return self.learning_rate
        """Get learning rate warmup."""
        print("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
              (hparams.learning_rate, WARMUP_STEPS, WARMUP_SCHEME))

        # Apply inverse decay if global steps less than warmup steps.
        # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
        # When step < warmup_steps,
        #   learing_rate *= warmup_factor ** (warmup_steps - step)
        if WARMUP_SCHEME == "t2t":
            # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
            warmup_factor = tf.exp(tf.log(0.01) / WARMUP_STEPS)
            inv_decay = warmup_factor ** (
                tf.to_float(WARMUP_STEPS - self._global_step))
        else:
            raise ValueError("Unknown warmup scheme %s" % WARMUP_SCHEME)

        return tf.cond(
            self._global_step < WARMUP_STEPS,
            lambda: inv_decay * self.learning_rate,
            lambda: self.learning_rate,
            name="learning_rate_warump_cond")

    def _get_learning_rate_decay(self):
        return self.learning_rate if self.epoch < self.hparams.learning_rate_start_decay_epoch else \
            tf.train.exponential_decay(
                self.learning_rate,
                (self.epoch - self.hparams.learning_rate_start_decay_epoch),
                self.hparams.learning_rate_decay_steps,
                self.hprams.learning_rate_decay_rate,
                staircase=True
            )

    def _build_graph(self):
        pass

    @property
    def global_step(self):
        return self.processed_inputs_count / self.batch_size

    @property
    def data_size(self):
        return (self._input_data_train.size if self.train_mode else self._input_data_test.size) \
               or (self.hparams.train_size if self.train_mode else self.hparams.eval_size)

    @property
    def epoch(self):
        return self.processed_inputs_count // self.data_size + 1

    @property
    def epoch_exact(self):
        return self.processed_inputs_count / self.data_size

    @property
    def epoch_progress(self):
        return (self.processed_inputs_count % self.data_size) // self.batch_size

    @property
    def step_size(self):
        return self.batch_size
