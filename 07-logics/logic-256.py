import numpy as np
import theano
from theano import tensor as T

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

class NNLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=None):
        w_bound = np.sqrt(6. / (n_in + n_out + 1))
        w_value = np.asarray(rng.uniform(size=(n_in, n_out), low=-w_bound, high=w_bound), dtype=theano.config.floatX)
        b_value = np.zeros((n_out), dtype=theano.config.floatX)

        # create shared variables
        self.W = theano.shared(w_value, borrow=True)
        self.b = theano.shared(b_value, borrow=True)
        self.params = [ self.W, self.b ]

        # output
        output = T.dot(input, self.W) + self.b
        if activation != None:
            output = activation(output)
        self.output = output

class MyLearn(object):
    def __init__(self):
        self.epoch = 0

    def build_model(self, data, learning_rate=0.0001, batch_size=256):
        # create a neural network
        rng = np.random.RandomState(7919)
        t_X = T.matrix(dtype=theano.config.floatX)
        t_y = T.vector(dtype=theano.config.floatX)

        layer1 = NNLayer(rng, t_X, 8, 256, T.nnet.relu)
        layer2 = NNLayer(rng, layer1.output, 256, 256, T.nnet.relu)
        layer3 = NNLayer(rng, layer2.output, 256, 1, T.nnet.relu)
        output = T.cast((layer3.output + 0.5), 'int32')

        cost = T.sum((layer3.output - t_y.reshape((batch_size, 1))) ** 2)
        params = layer3.params + layer2.params + layer1.params
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
        print("{}: {}".format(self.epoch, gd_cost))
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
