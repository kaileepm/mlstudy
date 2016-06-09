import numpy as np
import theano
from theano import tensor as T

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from shared.my_ml import *

class LogicData(object):
    def __init__(self):
        data_X = np.zeros((256, 8), dtype=theano.config.floatX)
        data_y = np.zeros((256,), dtype=theano.config.floatX)

        # build X, y
        for i in range(256):
            for j in range(8):
                data_X[i][j] = (i >> j) & 1
            data_y[i] = 0 if np.random.rand() < 0.5 else 1

        # create shared tensors of (X, y)
        self.data_X = theano.shared(data_X)
        self.data_y = theano.shared(data_y)

        self.n_data = 4
        self.test_X = data_X
        self.test_y = data_y

class MyLearn(object):
    def __init__(self):
        self.epoch = 0

    def build_model(self, data, batch_size=256):
        # create a neural network
        rng = np.random.RandomState(7919)
        t_X = T.matrix(dtype=theano.config.floatX)
        t_y = T.vector(dtype=theano.config.floatX)
	t_learning_rate = T.scalar(dtype=theano.config.floatX)

        layer1 = NeuralNetworkLayer(rng, t_X, 8, 256, T.nnet.relu)
        layer2 = NeuralNetworkLayer(rng, layer1.output, 256, 256, T.nnet.relu)
        layer3 = NeuralNetworkLayer(rng, layer2.output, 256, 1, T.nnet.relu)
        output = T.cast((layer3.output + 0.5), 'int32')

        cost = T.sum((layer3.output - t_y.reshape((batch_size, 1))) ** 2)
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

        # evaluation function
        self.forward = theano.function([ t_X ], output)

    def train_once(self, learning_rate):
        self.epoch += 1
        gd_cost = self.train_model(learning_rate)
        print("{}: {}".format(self.epoch, gd_cost))
        return gd_cost

    def train_until_converge(self, target_cost):
	learning_rate = 1. / 4096
	print("Learning rate is {}".format(learning_rate))
        old_cost = self.train_once(learning_rate)

        while True:
            new_cost = self.train_once(learning_rate)
            if new_cost > old_cost:
                # overshooting
                print("Adjust learning rate")
	        learning_rate /= 2.
		print("Learning rate is {}".format(learning_rate))
            if new_cost < target_cost:
                # print("Met target")
                break
            elif new_cost == old_cost:
                # print("Converged")
                break

            old_cost = new_cost

        return new_cost

    def predict(self, data_X):
        return self.forward(data_X)

    def test(self, data):
        y = self.predict(data.test_X)
        print(y.reshape(1, 256))

def main():
    cost_target = 0.001

    i = 0
    data = []
    mmls = []

    print("Creating NN")
    data.append(LogicData())
    mmls.append(MyLearn())
    mmls[i].build_model(data[i])

    print("Training...")
    cost = mmls[i].train_until_converge(cost_target)
    if cost >= cost_target:
        print("  Failed")
        
    print("Original")
    print(data[0].test_y)

    print("Test result")
    mmls[i].test(data[i])

if __name__ == "__main__":
    main()
