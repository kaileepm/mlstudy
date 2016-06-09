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
    def __init__(self, logic):
        # find logic handler
        if logic == 'and':
            logic = self.logic_and
        elif logic == 'or':
            logic = self.logic_or
        elif logic == 'xor':
            logic = self.logic_xor
        elif logic == 'not':
            logic = self.logic_not
        elif logic == 'nand':
            logic = self.logic_nand
        elif logic == 'nor':
            logic = self.logic_nor
        elif logic == 'xnor':
            logic = self.logic_xnor
        else:
            print("Unknown logic.")
            return

        # input: 4x2 matrix
        data_X = np.zeros((4, 2), dtype=theano.config.floatX)
        # output: results of logic operation. array of 4
        data_y = np.zeros((4,), dtype=theano.config.floatX)

        # build X, y
        i = 0
        for d1 in range(2):
            for d2 in range(2):
                data_X[i][0] = d1
                data_X[i][1] = d2
                data_y[i] = logic(d1, d2)
                i += 1

        # create shared tensors of (X, y)
        self.data_X = theano.shared(data_X)
        self.data_y = theano.shared(data_y)

        self.n_data = 4
        self.test_X = data_X
        self.test_y = data_y

    def logic_and(self, d1, d2):
        return d1 & d2
    def logic_or(self, d1, d2):
        return d1 | d2
    def logic_xor(self, d1, d2):
        return d1 ^ d2
    def logic_not(self, d1, d2):
        return 0 if d1 else 1
    def logic_nand(self, d1, d2):
        return 0 if (d1 & d2) else 1
    def logic_nor(self, d1, d2):
        return 0 if (d1 | d2) else 1
    def logic_xnor(self, d1, d2):
        return 0 if (d1 ^ d2) else 1

class MyLearn(object):
    def __init__(self):
        self.epoch = 0

    def build_model(self, data, learning_rate=0.01, batch_size=4):
        # create a neural network
        rng = np.random.RandomState(37)
        t_X = T.matrix(dtype=theano.config.floatX)
        t_y = T.vector(dtype=theano.config.floatX)

        layer1 = NeuralNetworkLayer(rng, t_X, 2, 2, T.nnet.relu)
        layer2 = NeuralNetworkLayer(rng, layer1.output, 2, 1, T.nnet.relu)
        output = T.cast((layer2.output + 0.5), 'int32')

        # training function
        # output of layer2: (N, 1)
        # t_y: (1, N)
        cost = T.sum((layer2.output - t_y.reshape((batch_size, 1))) ** 2)
        params = layer2.params + layer1.params
        grads = T.grad(cost, params)

        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        self.train_model = theano.function(
            [ ],
            cost,
            updates=updates,
            givens={
                t_X: data.data_X,
                t_y: data.data_y,
            }
        )

        # evaluation function
        self.forward = theano.function([ t_X ], output)

    def train_once(self):
        self.epoch += 1
        gd_cost = self.train_model()
        # print("{}: {}".format(self.epoch, gd_cost))
        return gd_cost

    def train_until_converge(self, target_cost):
        old_cost = self.train_once()

        while True:
            new_cost = self.train_once()
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
        print(y)

def main():
    cost_target = 0.001

    logics = ( 'and', 'or', 'xor', 'not', 'nand', 'nor', 'xnor' ) 
    data = []
    mmls = []

    for i in range(len(logics)):
        print("Creating NN for {}".format(logics[i]))
        data.append(LogicData(logics[i]))
        mmls.append(MyLearn())
        mmls[i].build_model(data[i])

        print("Training...")
        cost = mmls[i].train_until_converge(cost_target)
        if cost >= cost_target:
            print("  Failed")
        
        print("Test result")
        mmls[i].test(data[i])

if __name__ == "__main__":
    main()
