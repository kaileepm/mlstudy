#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from shared.my_ml import *

#
# FizzBuzz Neural Network + Multi-class Logistic Regression
#

class MyData(object):
    def __init__(self):
        n_i_begin = 100
        n_bits = 10
        n_data = (1 << n_bits) - n_i_begin
        n_class = 4

        # data_X: matrix with shape (n_data, n_bits)
        # data_y: vector (n_data)
        data_X = np.zeros((n_data, n_bits), dtype=theano.config.floatX)
        data_y = np.zeros((n_data,), dtype=theano.config.floatX)

        # build X, y
        i = 0
        for idx in range(n_i_begin, 1 << n_bits):
            for j in range(n_bits):
                data_X[i][j] = (i >> j) & 1
            if i % 15 == 0:
                data_y[i] = 3
            elif i % 3 == 0:
                data_y[i] = 2
            elif i % 5 == 0:
                data_y[i] = 1
            else:
                data_y[i] = 0
            i += 1

        # create shared tensors of (X, y)
        self.data_X = theano.shared(np.asarray(data_X, dtype=theano.config.floatX), borrow=True)
        self.data_y_float = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
        self.data_y = T.cast(self.data_y_float, 'int32')

        # save information
        self.np_data_X = data_X
        self.np_data_y = data_y
        self.n_bits = n_bits
        self.n_data = n_data
        self.n_class = n_class

        # test data
        self.test_data_X = np.zeros((100, n_bits), dtype=theano.config.floatX)
        self.test_data_y = np.zeros((100,), dtype=theano.config.floatX)
        for i in range(100):
            for j in range(n_bits):
                self.test_data_X[i][j] = (i >> j) & 1
            if i % 15 == 0:
                self.test_data_y[i] = 3
            elif i % 3 == 0:
                self.test_data_y[i] = 2
            elif i % 5 == 0:
                self.test_data_y[i] = 1
            else:
                self.test_data_y[i] = 0

class MyLearn(object):
    def __init__(self):
        self.epoch = 0

    def build_model(self, data, batch_size=256):
        # create a neural network
        rng = np.random.RandomState(7919)
        t_X = T.matrix(dtype=theano.config.floatX)
        t_y = T.ivector()
        t_learning_rate = T.scalar(dtype=theano.config.floatX)

        layer1 = NeuralNetworkLayer(rng, t_X, data.n_bits, 256, T.nnet.relu)
        layer2 = NeuralNetworkLayer(rng, layer1.output, 256, 256, T.nnet.relu)
        layer3 = LogisticRegressionMultiClass(layer2.output, 256, data.n_class)

        # create training model
        cost = layer3.negative_log_likelihood(t_y)
        params = layer3.params + layer2.params + layer1.params
        grads = T.grad(cost, params)

        updates = [
            (param_i, param_i - t_learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        self.train_model = theano.function(
            [ t_learning_rate ],
            cost,
            updates=updates,
            givens={
                t_X: data.data_X,
                t_y: data.data_y,
            }
        )

        # forward model
        self.forward_model = theano.function([ t_X ], layer3.output)

        # validation
        self.test_model = theano.function([], layer3.errors(t_y),
            givens={
                t_X: data.data_X,
                t_y: data.data_y,
            }
        )

    def validate(self):
        errors = self.test_model()
        print("Validation error: {}".format(errors))

    def predict(self, data_X):
        return self.forward_model(data_X)

    def train_once(self, learning_rate):
        if self.epoch % 100 == 0:
            self.validate()
        self.epoch += 1
        gd_cost = self.train_model(learning_rate)
        # print("{}: {}".format(self.epoch, gd_cost))
        return gd_cost

    def train_until_converge(self, target_cost):
        learning_rate = 1. / 8
        print("Learning rate is {}".format(learning_rate))
        old_cost = self.train_once(learning_rate)

        while True:
            new_cost = self.train_once(learning_rate)
            if new_cost > old_cost:
                # overshooting
                # print("Adjust learning rate")
                # learning_rate /= 2.
                # print("Learning rate is {}".format(learning_rate))
                None
            if new_cost < target_cost:
                # print("Met target")
                break
            elif new_cost == old_cost:
                # print("Converged")
                break

            old_cost = new_cost

        return new_cost

def main():
    cost_target = 0.001

    print("Creating data")
    data = MyData()

    print("Creating NN")
    mml = MyLearn()
    mml.build_model(data, batch_size = data.n_data)

    print("Training...")
    cost = mml.train_until_converge(cost_target)
    if cost >= cost_target:
        print("  Failed")

    print("Validating...")
    mml.validate()

    print("Test...")
    y = mml.predict(data.test_data_X)

    fizzbuzz = [ "", "fizz", "buzz", "fizzbuzz" ]
    wrong = 0
    for i in range(100):
        print("{}: {} {}".format(i, i if y[i] == 0 else fizzbuzz[y[i]], "WRONG" if y[i] != data.test_data_y[i] else ""))
        if y[i] != data.test_data_y[i]:
            wrong += 1

    print("Got {} wrong answers.".format(wrong))

if __name__ == "__main__":
    main()
