import argparse
import os
import tensorflow as tf
import random
import numpy as np
import sys
import shutil
from . import configs
from .utils import utils, ops_utils, model_utils
from tqdm import tqdm
from tensorflow.python import debug as tf_debug

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)
Model, InputData = None, None


class Trainer(object):
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
            name="batch_size")
        self.eval_batch_size = self.hparams.eval_batch_size
        self._eval_batch_size = tf.Variable(
            self.eval_batch_size,
            trainable=False, name="eval_batch_size")

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
            self.eval_batch_size,
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
                    self._input_data_train)
                opt = utils.get_optimizer(self.hparams, self.learning_rate)
                self.loss = model.loss
                self.params = model.trainable_variables()

                # compute gradient & update variables
                gradients = opt.compute_gradients(
                    self.loss,
                    var_list=self.params,
                    colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)

                clipped_grads, grad_norm_summary, grad_norm = model_utils.gradient_clip(
                    [grad for grad, _ in gradients], max_gradient_norm=self.hparams.max_gradient_norm)
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
                    self._input_data_test)
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
        ground_truth_labels, predicted_labels, ground_truth_len, predicted_len, loss, summary, extra_ops = \
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
                    if acc_id not in lers: lers[acc_id] = []
                    for i in range(len(gt_labels)):
                        ler, _, _ = ops_utils.evaluate(
                            gt_labels[i],#[:gt_len[i]],
                            p_labels[i],#[:p_len[i]],
                            decode_fns[acc_id],
                            metrics[acc_id])
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
                self.hprams.learning_rate_decay_rate, staircase=True)

    def _build_graph(self):
        pass

    def load(self, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        # if flags.load_ignore_scope:
        #    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=FLAGS.load_ignore_scope):
        #        saver_variables.remove(var)

        var_list = {var.op.name: var for var in saver_variables}

        var_map = {
            # "decoder/decoder_emb_layer/kernel": "decoder/dense/kernel",
            # "decoder/decoder_emb_layer/bias": "decoder/dense/bias",
            # "decoder/decoder_emb_layer/bias/Adam": "decoder/dense/bias/Adam",
            # "decoder/decoder_emb_layer/bias/Adam_1": "decoder/dense/bias/Adam_1",
            # "decoder/decoder_emb_layer/kernel/Adam": "decoder/dense/kernel/Adam",
            # "decoder/decoder_emb_layer/kernel/Adam_1": "decoder/dense/kernel/Adam_1",
        }

        # fine-tuning for context
        # print(var_list)
        # del var_list["decoder/attention_wrapper/basic_lstm_cell/kernel"]

        for it in var_map:
            var_list[var_map[it]] = var_list[it]
        # del var_list[it]

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

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


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--verbose', type="bool", const=True, nargs="?", default=False)
    parser.add_argument('--reset', type="bool", const=True, nargs="?", default=False,
                        help="No saved model loaded")
    parser.add_argument('--debug', type="bool", const=True, nargs="?", default=False)
    parser.add_argument('--eval', type=int, default=500,
                        help="Frequently check and log evaluation result")
    parser.add_argument('--eval_from', type=int, default=0,
                        help="No. of epoch before eval")
    parser.add_argument('--transfer', type="bool", const=True, nargs="?", default=False,
                        help="If model needs custom load.")

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--output', type="bool", const=True, nargs="?", default=False)
    parser.add_argument('--simulated', type="bool", const=True, nargs="?", default=False)
    parser.add_argument('--model', type=str, default="ctc-attention")
    parser.add_argument('--input_unit', type=str, default="char", help="word | char")

    parser.add_argument('--load_ignore_scope', type=str, default=None)

    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")


def load_model(sess, Model, hparams):
    sess.run(tf.global_variables_initializer())
    if FLAGS.reset: return
    if FLAGS.load:
        ckpt = os.path.join(hparams.out_dir, "csp.%s.ckpt" % FLAGS.load)
    else:
        ckpt = tf.train.latest_checkpoint(hparams.out_dir)

    if not ckpt: return

    if FLAGS.transfer:
        Model.load(sess, ckpt, FLAGS)
    else:
        saver_variables = tf.global_variables()
        var_list = {var.op.name: var for var in saver_variables}
        for var in Model.ignore_save_variables():
            if var in var_list:
                del var_list[var]
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)


def save(hparams, sess, name=None):
    path = os.path.join(hparams.out_dir, "csp.%s.ckpt" % (name))
    saver = tf.train.Saver()
    saver.save(sess, path)
    if hparams.verbose: print("Saved as csp.%s.ckpt" % (name))


def argval(name):
    return utils.argval(name, FLAGS)


def train(Model, BatchedInput, hparams):
    hparams.beam_width = 0
    graph = tf.Graph()
    mode = tf.estimator.ModeKeys.TRAIN
    with graph.as_default():
        # new_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder/correction')
        # init_new_vars = tf.initialize_variables(new_vars)

        trainer = Trainer(hparams, Model, BatchedInput, mode)
        trainer.build_model(eval=argval("eval") != 0)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sess = tf.Session(graph=graph, config=config)
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        load_model(sess, Model, hparams)

        if argval("simulated"):
            # not real training, only to export values
            utils.prepare_output_path(hparams)
            sess.run(tf.assign(trainer._global_step, 0))
            sess.run(tf.assign(trainer._processed_inputs_count, 0))

        trainer.init(sess)

        # tensorboard log
        if FLAGS.reset:
            if os.path.exists(hparams.summaries_dir):
                shutil.rmtree(hparams.summaries_dir)
        writer = tf.summary.FileWriter(os.path.join(hparams.summaries_dir), sess.graph)

        last_save_step = trainer.global_step
        last_eval_pos = trainer.global_step - FLAGS.eval

        def reset_pbar():
            epoch_batch_count = trainer.data_size // trainer.step_size
            pbar = tqdm(total=epoch_batch_count,
                        ncols=150,
                        unit="step",
                        initial=trainer.epoch_progress)
            pbar.set_description('Epoch %i' % trainer.epoch)
            return pbar

        pbar = reset_pbar()
        last_epoch = trainer.epoch
        dev_lers = {}
        min_dev_lers = {}
        test_lers = {}
        min_test_lers = {}
        min_dev_test_lers = {}

        trainer.reset_train_iterator(sess)

        while True:
            # utils.update_hparams(FLAGS, hparams) # renew hparams so paramters can be changed during training

            # eval if needed
            if argval("eval") > 0 and argval("eval_from") <= trainer.epoch_exact:
                if trainer.global_step - last_eval_pos >= FLAGS.eval:
                    pbar.set_postfix_str("Evaluating (dev)...")
                    dev_lers = trainer.eval_all(sess, dev=True)
                    pbar.set_postfix_str("Evaluating (test)...")
                    test_lers = trainer.eval_all(sess, dev=False)

                    for acc_id in test_lers:
                        if dev_lers is None:
                            if acc_id not in min_test_lers or min_test_lers[acc_id] > test_lers[acc_id]:
                                min_test_lers[acc_id] = test_lers[acc_id]
                                save(hparams, sess, "best_%d" % acc_id)
                        else:
                            if acc_id not in min_test_lers or min_test_lers[acc_id] > test_lers[acc_id]:
                                min_test_lers[acc_id] = test_lers[acc_id]

                            if acc_id not in min_dev_lers or (min_dev_lers[acc_id] > dev_lers[acc_id]):
                                min_dev_lers[acc_id] = dev_lers[acc_id]
                                min_dev_test_lers[acc_id] = test_lers[acc_id]
                                save(hparams, sess, "best_%d" % acc_id)

                            tqdm.write("dev: %.2f, test: %.2f, acc: %.2f" %
                                       (dev_lers[acc_id] * 100, test_lers[acc_id] * 100, min_test_lers[acc_id] * 100))

                        for (err_id, lers) in [("dev", dev_lers), ("test", test_lers), ("min_test", min_dev_test_lers)]:
                            if lers is not None and len(lers) > 0:
                                writer.add_summary(
                                    tf.Summary(value=[tf.Summary.Value(simple_value=lers[acc_id],
                                                                       tag="%s_error_rate_%d" % (err_id, acc_id))]),
                                    trainer.processed_inputs_count)

                    last_eval_pos = trainer.global_step

            loss, summary = trainer.train(sess)

            # return

            if trainer.epoch > last_epoch:  # reset epoch
                pbar = reset_pbar()
                last_epoch = trainer.epoch

            writer.add_summary(summary, trainer.processed_inputs_count)
            pbar.update(1)

            if not argval("simulated") and trainer.global_step - last_save_step >= FLAGS.save_steps:
                save(hparams, sess, "epoch%d" % trainer.epoch)
                last_save_step = trainer.global_step

            if trainer.epoch > hparams.max_epoch_num: break

            # reduce batch size with long input
            if hparams.batch_size_decay:
                if trainer.decay_batch_size(trainer.epoch_exact -
                                            trainer.epoch, sess):
                    pbar = reset_pbar()

            # update postfix
            pbar_pf = {}
            for acc_id in test_lers:
                if dev_lers is not None: pbar_pf["min_dev" + str(acc_id)] = "%2.2f" % (min_dev_test_lers[acc_id] * 100)
                pbar_pf["min_test" + str(acc_id)] = "%2.2f" % (min_test_lers[acc_id] * 100)
                pbar_pf["test" + str(acc_id)] = "%2.2f" % (test_lers[acc_id] * 100)
                if dev_lers is not None: pbar_pf["dev" + str(acc_id)] = "%2.2f" % (dev_lers[acc_id] * 100)
            pbar_pf['cost'] = "%.3f" % (loss)
            pbar.set_postfix(pbar_pf)


def main(unused_argv):
    global Model, InputData

    Model, InputData, hparams = utils.read_configs(FLAGS)
    hparams.hcopy_path = configs.HCOPY_PATH
    hparams.hcopy_config = os.path.join(configs.HCOPY_CONFIG_PATH)

    if not os.path.exists(hparams.summaries_dir): os.mkdir(hparams.summaries_dir)
    if not os.path.exists(hparams.out_dir): os.mkdir(hparams.out_dir)
    utils.clear_log(hparams)

    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    train(Model, InputData, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
