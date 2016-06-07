#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import cPickle
import gzip

class LogisticRegression(object):
    def __init__(self, t_data_x, n_x, n_y):
        # Weights
        self.W = theano.shared(np.zeros((n_x, n_y), dtype='float64'), borrow=True)
        # Bias
        self.b = theano.shared(np.zeros((n_y), dtype='float64'), borrow=True)
        # the probability that an input vector x is a member of class i
        # P(Y = i|x,W,b) = softmax(Wx + b)
        self.p_y_given_x = T.nnet.softmax(T.dot(t_data_x, self.W) + self.b)
        # prediction is the class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, t_data_y):
        # cost function
        return -T.mean(T.log(self.p_y_given_x)[T.arange(t_data_y.shape[0]), t_data_y])

    def errors(self, t_data_y):
        # error rates
        return T.mean(T.neq(self.y_pred, t_data_y))

class MNISTData:
    def __init__(self):
        self.n_x = None

    def make_shared(self, data_set):
        data_x, data_y = data_set

        # make shared variables for GPU.
        # for GPU, the data type should be float
        shared_data_x = theano.shared(np.asarray(data_x, dtype='float64'))
        shared_data_y = theano.shared(np.asarray(data_y, dtype='float64'))

        # for calculation, y should be integer
        return shared_data_x, T.cast(shared_data_y, 'int32')

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
        self.k = 10

        self.n_train = len(train_set[0])
        self.n_valid = len(valid_set[0])
        self.n_test = len(test_set[0])

        print("  Training set: {}".format(train_set[0].shape))
        print("  Validation set: {}".format(valid_set[0].shape))
        print("  Test set: {}".format(test_set[0].shape))

        #
        # Create shared variables to make use of GPUs
        #

        self.train_set_x, self.train_set_y = self.make_shared(train_set)
        self.valid_set_x, self.valid_set_y = self.make_shared(valid_set)
        self.test_set_x, self.test_set_y = self.make_shared(test_set)

        return True

class MyLearn:
    # Logistic regression for multiple classification
    # Mini-batch Stochastic Gradient Descent

    def __init__(self, data):
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

        self.classifier = LogisticRegression(t_data_x, self.data.n_x, self.data.k)
        cost = self.classifier.negative_log_likelihood(t_data_y)

        grad_W = T.grad(cost, self.classifier.W)
        grad_b = T.grad(cost, self.classifier.b)

        self.train_model = theano.function([t_batch_index], cost,
            updates=[ (self.classifier.W, self.classifier.W - learning_rate * grad_W),
                      (self.classifier.b, self.classifier.b - learning_rate * grad_b) ],
            givens={ t_data_x: self.data.train_set_x[t_batch_index * batch_size : (t_batch_index + 1) * batch_size],
                     t_data_y: self.data.train_set_y[t_batch_index * batch_size : (t_batch_index + 1) * batch_size] })

        #
        # Create testing (validation, test) models
        #

        self.valid_model = theano.function([], self.classifier.errors(self.data.valid_set_y),
            givens={ t_data_x: self.data.valid_set_x, t_data_y: self.data.valid_set_y })

        self.test_model = theano.function([], self.classifier.errors(self.data.test_set_y),
            givens={ t_data_x: self.data.test_set_x, t_data_y: self.data.test_set_y })

    def train_once(self):
        batch_count = self.data.n_train / self.batch_size

        for batch_index in range(batch_count):
            gd_cost = self.train_model(batch_index)

        return gd_cost

    def train_until_converge(self, target_diff):
        epoch = 1
        old_cost = self.train_once()

        while True:
            new_cost = self.train_once()
            print("{}: {}".format(epoch, new_cost))

            if old_cost - new_cost < target_diff:
                print("Converged")
                break

            epoch += 1
            old_cost = new_cost

    def train_multiple(self, n_epoch):
        for epoch in range(n_epoch):
            gd_cost = self.train_once()
            print("{}: {}".format(epoch, gd_cost))

    def validate(self):
        errors = self.valid_model()
        print("Validation: {}".format(errors))

    def test(self):
        errors = self.test_model()
        print("Test: {}".format(errors))

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
    if False:
        for i in range(10):
            mm.train_multiple(10)
            mm.validate()

    if True:
        mm.train_until_converge(0.0001)
        mm.validate()

    print("Testing...")
    mm.test()

if __name__ == "__main__":
    main()
