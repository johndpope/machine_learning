#!/bin/env python
# coding:utf-8
# http://qiita.com/chachay/items/052406176c55dd5b9a6a
# http://qiita.com/mitmul/items/eccf4e0a84cb784ba84a
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import matplotlib.pyplot as plt
import numpy
numpy.random.seed(0)
import chainer
if chainer.cuda.available:
    chainer.cuda.cupy.random.seed(0)

class MLP(Chain):
    n_input  = 1
    n_output = 1
    n_units  = 5

    def __init__(self):
        super(MLP, self).__init__(
            l1 = L.Linear(self.n_input, self.n_units),
            l2 = L.LSTM(self.n_units, self.n_units),
            l3 = L.Linear(self.n_units, self.n_output),
        )

    def reset_state(self):
        self.l2.reset_state()

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        return self.l3(h2)

def data_set():
    # データ作成
    N_data = 100
    N_Loop = 3
    t = np.linspace(0., 2 * np.pi * N_Loop, num=N_data)

    X = 0.8 * np.sin(2.0 * t)

    # データセット
    N_train = int(N_data * 0.8)
    N_test = int(N_data * 0.2)

    tmp_DataSet_X = np.array(X).astype(np.float32)

    x_train, x_test = np.array(tmp_DataSet_X[:N_train]), np.array(tmp_DataSet_X[N_train:])

    train = tuple_dataset.TupleDataset(x_train)
    test = tuple_dataset.TupleDataset(x_test)
    return(train,test)



class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size

        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        self.offsets = [i * length // batch_size for i in range(batch_size)]

        self.iteration = 0

    def __next__(self):

        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)



batchsize=10
gpu=-1
epoch=20
out="log"
frequency=10

rnn=MLP()
model = L.Classifier(rnn)
optimizer = optimizers.Adam()
optimizer.setup(model)

train,test=data_set()

train_iter = ParallelSequentialIterator(train, batchsize)
test_iter = ParallelSequentialIterator(test, batchsize)

# Set up a trainer
updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot for each specified epoch
frequency = epoch if frequency == -1 else max(1, frequency)
trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Save two plot images to the result dir
if extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'))

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

# Run the training
trainer.run()
chainer.serializers.save_npz("sin.trainer", trainer)
