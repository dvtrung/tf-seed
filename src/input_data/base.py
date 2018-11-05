import os
import random
from struct import unpack
from subprocess import call, PIPE

import numpy as np
import tensorflow as tf
from ..utils import utils


class BaseInputData():
    CLASSES = [str(i) for i in range(9)]
    DEFAULT_PARAMS = {}

    def __init__(self,
                 params,
                 mode,
                 batch_size,
                 dev=False):
        self._params = params
        self._mode = mode
        self._batch_size = tf.cast(batch_size, tf.int64)

    def get_batched_dataset(self, dataset):
        return utils.get_batched_dataset(
            dataset,
            self.batch_size,
            self.hparams.num_features,
            self.mode,
            padding_values=self.hparams.eos_index
        )

    @property
    def iterator(self): return self._iterator

    def load_wav(self, filename):
        outfile = "tmp.htk"
        call([
            self.hparams.hcopy_path,
            "-C", self.hparams.hcopy_config, "-T", "1",
            filename, outfile
        ], stdout=PIPE)
        fh = open(outfile, "rb")
        spam = fh.read(12)
        # print("spam", spam)
        nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
        veclen = int(sampSize / 4)
        fh.seek(12, 0)
        dat = np.fromfile(fh, dtype=np.float32)
        dat = dat.reshape(len(dat) // veclen, veclen)
        dat = dat.byteswap()
        fh.close()

        dat = (dat - self.mean) / np.sqrt(self.var)
        fh.close()

        return np.float32(dat)

    def load_htk(self, filename):
        fh = open(filename, "rb")
        spam = fh.read(12)
        nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
        veclen = int(sampSize / 4)
        fh.seek(12, 0)
        dat = np.fromfile(fh, dtype=np.float32)
        if len(dat) % veclen != 0: dat = dat[:len(dat) - len(dat) % veclen]
        dat = dat.reshape(len(dat) // veclen, veclen)
        dat = dat.byteswap()
        fh.close()
        return dat

    def load_npy(self, filename):
        #return np.array([[0.0]], dtype=np.float32)
        dat = np.load(filename.decode('utf-8')).astype(np.float32)
        return dat

    def load_input(self, filename):
        if os.path.splitext(filename)[1] == b".htk":
            return self.load_htk(filename)
        elif os.path.splitext(filename)[1] == b".wav":
            return self.load_wav(filename)
        elif os.path.splitext(filename)[1] == b".npy":
            return self.load_npy(filename)
        else:
            return np.array([[0.0] * 120] * 8).astype(np.float32)

    def decode(self, d):
        """Decode from label ids to words"""
        return self.CLASSES[d]

    def shuffle(self, inputs, bucket_size=None):
        if bucket_size:
            shuffled_inputs = []
            for i in range(0, len(inputs) // bucket_size):
                start, end = i * bucket_size, min((i + 1) * bucket_size, len(inputs))
                ls = inputs[start:end]
                random.shuffle(ls)
                shuffled_inputs += ls
            return shuffled_inputs
        else:
            ls = list(inputs)
            random.shuffle(ls)
            return ls

    def get_inputs_list(self, inputs, field):
        return [inp[field] for inp in self.inputs]

    def init_dataset(self):
        pass

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        pass
