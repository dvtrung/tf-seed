import argparse
import os
import sys

import tensorflow as tf
from .helpers.trainer import TrainerHelper
from tqdm import tqdm

from . import configs
from .utils import utils, ops_utils

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)

Model = InputData = None


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--mode', type=str, default="eval")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--metrics', type=str, default=None)

    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--output', type="bool", const=True, nargs="?", default=False)

    parser.add_argument("--summaries_dir", type=str, default="log")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")


def load_model(sess, Model, hparams):
    sess.run(tf.global_variables_initializer())

    if hparams.load:
        ckpt = os.path.join(hparams.out_dir, "csp.%s.ckpt" % hparams.load)
    else:
        ckpt = tf.train.latest_checkpoint(hparams.out_dir)

    if ckpt:
        saver_variables = tf.global_variables()
        var_list = {var.op.name: var for var in saver_variables}
        for var in Model.ignore_save_variables() + ["batch_size",
                                                    "eval_batch_size", 'Variable_1']:
            if var in var_list:
                del var_list[var]
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)


def eval(hparams, args):
    tf.reset_default_graph()
    graph = tf.Graph()
    mode = tf.estimator.ModeKeys.EVAL
    hparams.batch_size = hparams.eval_batch_size

    global Model, InputData

    with graph.as_default():
        trainer = TrainerHelper(hparams, Model, InputData, mode)
        trainer.build_model()

        sess = tf.Session(graph=graph)
        load_model(sess, Model, hparams)
        trainer.init(sess)

        dlgids = []
        lers = []

        pbar = tqdm(total=trainer.data_size, ncols=100)
        pbar.set_description("Eval")
        fo = open(os.path.join(hparams.summaries_dir, "eval_ret.txt"), "w")
        utils.prepare_output_path(hparams)
        lers = {}
        while True:
            try:
                ground_truth_labels, predicted_labels, ground_truth_len, predicted_len = trainer.eval(sess)
                utils.write_log(hparams, [str(ground_truth_labels)])

                decode_fns = Model.get_decode_fns()
                # dlgids += list([str(id).split('/')[-2] for id in ids])
                metrics = (args.metrics or hparams.metrics).split(',')
                for acc_id, (gt_labels, p_labels, gt_len, p_len) in \
                        enumerate(zip(ground_truth_labels, predicted_labels,
                                      ground_truth_len, predicted_len)):
                    if acc_id not in lers: lers[acc_id] = []
                    for i in range(len(gt_labels)):
                        if acc_id == 1 and (hparams.model == "da_attention_seg"):
                            ler, str_original, str_decoded = ops_utils.joint_evaluate(
                                hparams,
                                ground_truth_labels[0][i], predicted_labels[0][i],
                                ground_truth_labels[1][i], predicted_labels[1][i],
                                decode_fns[acc_id],
                            )
                        else:
                            ler, str_original, str_decoded = ops_utils.evaluate(
                                # gt_labels[i],
                                gt_labels[i],#[:gt_len[i]],
                                # p_labels[i],
                                p_labels[i],#[:p_len[i]],
                                decode_fns[acc_id],
                                metrics[acc_id]
                            )
                        if ler is not None:
                            lers[acc_id].append(ler)

                            tqdm.write(
                                "\nGT: %s\nPR: %s\nLER: %.3f\n" % (' '.join(str_original), ' '.join(str_decoded), ler))

                            meta = tf.SummaryMetadata()
                            meta.plugin_data.plugin_name = "text"

                # update pbar progress and postfix
                pbar.update(trainer.batch_size)
                bar_pf = {}
                for acc_id in range(len(ground_truth_labels)):
                    bar_pf["er" + str(acc_id)] = "%2.2f" % (sum(lers[acc_id]) / len(lers[acc_id]) * 100)
                pbar.set_postfix(bar_pf)
            except tf.errors.OutOfRangeError:
                break

    # acc_by_ids = {}
    # for i, id in enumerate(dlgids):
    #    if id not in acc_by_ids: acc_by_ids[id] = []
    #    acc_by_ids[id].append(lers[0][i])

    # print("\n\n----- Statistics -----")
    # for id, ls in acc_by_ids.items():
    #     print("%s\t%2.2f" % (id, sum(ls) / len(ls)))

    # fo.write("LER: %2.2f" % (sum(lers) / len(lers) * 100))
    # print(len(lers[0]))
    fo.close()


def main(unused_argv):
    global Model, InputData
    Model, InputData, hparams = utils.read_configs(args)
    hparams.hcopy_path = configs.HCOPY_PATH
    hparams.hcopy_config = configs.HCOPY_CONFIG_PATH
    if not os.path.exists(hparams.summaries_dir): os.mkdir(hparams.summaries_dir)
    eval(hparams, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])
