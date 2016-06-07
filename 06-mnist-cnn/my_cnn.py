#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.signal import downsample
import cPickle
import gzip

class LogisticRegression(object):
    def __init__(self, t_data_x, n_x, n_y):
        # Weights
        self.W = theano.shared(np.zeros((n_x, n_y), theano.config.floatX), borrow=True)
        # Bias
        self.b = theano.shared(np.zeros((n_y), dtype=theano.config.floatX), borrow=True)
        # the probability that an input vector x is a member of class i
        # P(Y = i|x,W,b) = softmax(Wx + b)
        self.p_y_given_x = T.nnet.softmax(T.dot(t_data_x, self.W) + self.b)
        # prediction is the class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # parameter list
        self.params = [ self.W, self.b ]

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
        shared_data_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        shared_data_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

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

class LeNetConvPoolLayer(object):
    # convolution in theano
    # conv2d()
    #   input, filters: dtensor3
    #   image_shape, filter_shape: ([number of images/filters,] height, width)

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        # rng: random number generator
        # input: batch of input images - dtensor4
        # filter_shape: (number of filters, number of input feature maps, height, width)
        # image_shape: (batch size, number of input feature maps, height, width)
        # poolsize: (# rows, # columns)

        # calculate the range of random number
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        # parameters
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=True)
        self.params = [ self.W, self.b ]

        # convolution layer
        conv_out = T.nnet.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # pooling layer
        pooled_out = T.signal.downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # activation function
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [ self.W, self.b ]

class MyLearn:
    # Logistic regression for multiple classification
    # Mini-batch Stochastic Gradient Descent

    def __init__(self, data):
        self.epoch = 0
        self.data = data

    def build_model(self, learning_rate=0.1, batch_size=500, nkerns=[20, 50]):
        #
        # CNN architecture
        #
        # input: 28x28x1
        # convolution (layer0): 24x24
        # max pooling (layer0): 12x12
        # activation (layer0)
        # convolution (layer1): 8x8
        # max pooling (layer1): 4x4
        # activation (layer1)
        # full connect (layer2): 500 outputs
        # full connect (layer3): logistic regression (softmax), 10 outputs
        #

        rng = np.random.RandomState(23455)
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

        layer0_input = t_data_x.reshape((batch_size, 1, 28, 28))

        # input: 28x28x1
        # filter: nkerns[0] 5x5
        # after convolution: 24x24xnkerns[0]
        # after pooling: 12x12xnkerns[0]
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )

        # input: 12x12xnkerns[0]
        # filter: nkerns[1] 5x5
        # after convolution: 8x8xnkerns[1]
        # after pooling: 4x4xnkerns[1]
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        layer2_input = layer1.output.flatten(2)

        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 4 * 4,
            n_out=500,
            activation=T.tanh
        )

        layer3 = LogisticRegression(layer2.output, 500, self.data.k)
        cost = layer3.negative_log_likelihood(t_data_y)

        params = layer3.params + layer2.params + layer1.params + layer0.params
        grads = T.grad(cost, params)

        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        self.train_model = theano.function(
            [t_batch_index],
            cost,
            updates=updates,
            givens={
                t_data_x: self.data.train_set_x[t_batch_index * batch_size: (t_batch_index + 1) * batch_size],
                t_data_y: self.data.train_set_y[t_batch_index * batch_size: (t_batch_index + 1) * batch_size]
            }
        )

        self.classifier = layer3

        #
        # Create testing (validation, test) models
        #
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
        for i in range(100):
            mm.train_multiple(1)
            mm.validate()
            mm.test()

    if False:
        mm.train_until_converge(0.0001)
        mm.validate()

    print("Testing...")
    mm.test()

if __name__ == "__main__":
    main()

