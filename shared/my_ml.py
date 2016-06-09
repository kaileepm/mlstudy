import numpy as np
import theano
from theano import tensor as T

#
# Multi-class Logistic Regression
#

class LogisticRegressionMultiClass(object):
    def __init__(self, t_X, n_in, n_out):
        # Arguments:
        #   t_X: input data as a tensor
        #   n_in: number of input units
        #   n_out: number of output units

        # Weights: a matrix of size n_in x n_out
        self.W = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), borrow=True)
        # Bias: a vector of size n_out
        self.b = theano.shared(np.zeros((n_out), dtype=theano.config.floatX), borrow=True)
        # List of parameters
        self.params = [ self.W, self.b ]

        # the probability that an input vector x is a member of class i
        # P(Y = i|x,W,b) = softmax(Wx + b)
        self.p_y_given_in = T.nnet.softmax(T.dot(t_X, self.W) + self.b)

        # prediction is the class whose probability is maximal
        self.output = T.argmax(self.p_y_given_in, axis=1)

    def negative_log_likelihood(self, t_y):
        # cost function
        # t_y.shape[0]: number of rows
        return -T.mean(T.log(self.p_y_given_in)[T.arange(t_y.shape[0]), t_y])

    def errors(self, t_y):
        # error rates
        return T.mean(T.neq(self.output, t_y))

    def debugprint(self, t_y):
        print('Function output:')
        theano.printing.debugprint(self.output)
        print('Function cost:')
        theano.printing.debugprint(self.negative_log_likelihood(t_y))
        print('Function error:')
        theano.printing.debugprint(self.errors(t_y))

#
# Neural Network Layer
#

class NeuralNetworkLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=None):
        # Arguments:
        #   rng: random number generator
        #   input: input data as a tensor
        #   n_in: number of input units
        #   n_out: number of output units
        #   activation: activation function
        #      None
        #      theano.tensor.nnet.sigmoid
        #      theano.tensor.tanh
        #      theano.tensor.nnet.relu

        # weight initialization
        # Xavier Glorot, Yoshua Bengio
        # normalized initialization: sqrt(6 / (n_in + n_out + 1))
        w_bound = np.sqrt(6. / (n_in + n_out + 1))
        w_value = np.asarray(rng.uniform(size=(n_in, n_out), low=-w_bound, high=w_bound), dtype=theano.config.floatX)

        # bias initialization
        # zeroed out
        b_value = np.zeros((n_out,), dtype=theano.config.floatX)

        # create shared variables
        self.W = theano.shared(w_value, borrow=True)
        self.b = theano.shared(b_value, borrow=True)
        self.params = [ self.W, self.b ]

        # output
        output = T.dot(input, self.W) + self.b
        if activation != None:
            output = activation(output)
        self.output = output

    def debugprint(self, t_y):
        print('Function output:')
        theano.printing.debugprint(self.output)
