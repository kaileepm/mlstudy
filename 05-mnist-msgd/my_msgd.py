#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import cPickle
import gzip

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from shared.my_ml import *

class MNISTData:
    def __init__(self):
        self.n_x = None

    def make_shared(self, data_set):
        data_x, data_y = data_set

        # make shared variables for GPU.
        # for GPU, the data type should be float
        shared_data_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        shared_data_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

        # for calculation, y should be integer
        return shared_data_x, T.cast(shared_data_y, 'int32'), data_x, data_y

    def load_data(self):
        #
        # Load
        #

        try:
            fd = gzip.open('../data/mnist.pkl.gz', 'rb')
            train_set, valid_set, test_set = cPickle.load(fd)
            fd.close()
        except:
            return False

        # the loaded data is a tuple of (x, y)
        # x: two dimensional array: number of training example * 784
        # 784: 28x28
        # classification: 10 possible outputs
        self.n_x = train_set[0].shape[1]
        self.n_class = 10

        self.n_train = len(train_set[0])
        self.n_valid = len(valid_set[0])
        self.n_test = len(test_set[0])

        print("  Training set: {}".format(train_set[0].shape))
        print("  Validation set: {}".format(valid_set[0].shape))
        print("  Test set: {}".format(test_set[0].shape))

        #
        # Create shared variables to make use of GPUs
        #

        self.train_set_x, self.train_set_y, self.train_np_x, self.train_np_y  = self.make_shared(train_set)
        self.valid_set_x, self.valid_set_y, self.valid_np_x, self.valid_np_y = self.make_shared(valid_set)
        self.test_set_x, self.test_set_y, self.test_np_x, self.test_np_y = self.make_shared(test_set)

        return True

class MyLearn:
    # Logistic regression for multiple classification
    # Mini-batch Stochastic Gradient Descent

    def __init__(self, data):
        self.epoch = 0
        self.data = data

    def build_model(self, learning_rate=0.1, batch_size=500):
        #
        # Create a training model
        #

        self.batch_size = batch_size

        #
        # tensor variables:
        #   t_batch_index
        #   t_data_x
        #   t_data_y
        #
        t_batch_index = T.iscalar()
        t_data_x = T.matrix()
        t_data_y = T.ivector()

        self.classifier = LogisticRegressionMultiClass(t_data_x, self.data.n_x, self.data.n_class)
        cost = self.classifier.negative_log_likelihood(t_data_y)

        params = self.classifier.params
        grads = T.grad(cost, params)

        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        if False: # initial implementation
            grad_W = T.grad(cost, self.classifier.W)
            grad_b = T.grad(cost, self.classifier.b)
            updates=[ (self.classifier.W, self.classifier.W - learning_rate * grad_W),
                (self.classifier.b, self.classifier.b - learning_rate * grad_b) ],

        self.train_model = theano.function([t_batch_index], cost, updates=updates,
            givens={ t_data_x: self.data.train_set_x[t_batch_index * batch_size : (t_batch_index + 1) * batch_size],
                     t_data_y: self.data.train_set_y[t_batch_index * batch_size : (t_batch_index + 1) * batch_size] })

        #
        # Create testing (validation, test) models
        #

        self.forward_model = theano.function([t_data_x], self.classifier.output)

        self.valid_model = theano.function([t_batch_index], self.classifier.errors(t_data_y),
            givens={ t_data_x: self.data.valid_set_x[t_batch_index * batch_size: (t_batch_index + 1) * batch_size],
                     t_data_y: self.data.valid_set_y[t_batch_index * batch_size: (t_batch_index + 1) * batch_size] })

        self.test_model = theano.function([t_batch_index], self.classifier.errors(t_data_y),
            givens={ t_data_x: self.data.test_set_x[t_batch_index * batch_size: (t_batch_index + 1) * batch_size],
                     t_data_y: self.data.test_set_y[t_batch_index * batch_size: (t_batch_index + 1) * batch_size] })

    def train_once(self):
        self.epoch += 1
        batch_count = self.data.n_train / self.batch_size

        for batch_index in range(batch_count):
            gd_cost = self.train_model(batch_index)

        return gd_cost

    def train_multiple(self, n_epoch):
        for epoch in range(n_epoch):
            gd_cost = self.train_once()
            print("{}: {}".format(self.epoch, gd_cost))

    def train_until_converge(self, target_diff):
        old_cost = self.train_once()
        print("{}: {}".format(self.epoch, old_cost))

        while True:
            new_cost = self.train_once()
            print("{}: {}".format(self.epoch, new_cost))

            if old_cost - new_cost < target_diff:
                print("Converged")
                break

            old_cost = new_cost

    def validate(self):
        batch_count = self.data.n_valid / self.batch_size

        errors = 0
        for batch_index in range(batch_count):
            errors += self.valid_model(batch_index)

        errors /= batch_count
        errors *= 100.
        print("Validation: {0:.2f}%".format(errors))

    def test(self):
        batch_count = self.data.n_test / self.batch_size

        errors = 0
        for batch_index in range(batch_count):
            errors += self.test_model(batch_index)

        errors /= batch_count
        errors *= 100.
        print("Test: {0:.2f}%".format(errors))

    def forward(self, data_X):
        return self.forward_model(data_X)

def main():
    print("Loading data...")
    data = MNISTData()
    if not data.load_data():
        print("Cannot load data.")
        return

    print("Building models...")
    mm = MyLearn(data)
    mm.build_model()

    print("Training...")
    if True:
        for i in range(10):
            mm.train_multiple(10)
            mm.validate()

    if True:
        mm.train_until_converge(0.0001)
        mm.validate()

    print("Testing...")
    mm.test()

    print("Testing forward...")
    y = mm.forward(data.test_np_x)
    print(y)

if __name__ == "__main__":
    main()
