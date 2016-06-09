import numpy as np
import theano
from theano import tensor as T

class MyData(object):
    def __init__(self):
        n_data = 1024
        data_X = np.zeros((n_data, 1), dtype=theano.config.floatX)
        data_y = np.zeros((n_data,), dtype=theano.config.floatX)

        # build X, y
        for i in range(n_data):
            data_X[i][0] = i
            if i % 15 == 0:
                data_y[i] = 3
            elif i % 3 == 0:
                data_y[i] = 1
            elif i % 5 == 0:
                data_y[i] = 2

        # create shared tensors of (X, y)
        self.data_X = theano.shared(np.asarray(data_X, dtype=theano.config.floatX), borrow=True)
        self.shared_data_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
        self.data_y = T.cast(self.shared_data_y, 'int32')

        # save information
        self.test_X = data_X
        self.test_y = data_y
        self.n_data = n_data
        self.k = 4

class NNLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=None):
        # weights and bias
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

class LogisticRegression(object):
    def __init__(self, t_X, n_x, n_y):
        self.W = theano.shared(np.zeros((n_x, n_y), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.zeros((n_y), dtype=theano.config.floatX), borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(t_X, self.W) + self.b)
        self.output = T.argmax(self.p_y_given_x, axis=1)
        self.params = [ self.W, self.b ]

    def negative_log_likelihood(self, t_y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(t_y.shape[0]), t_y])

    def errors(self, t_y):
        return T.mean(T.neq(self.output, t_y))

class MyLearn(object):
    def __init__(self):
        self.epoch = 0

    def build_model(self, data, batch_size=256):
        # create a neural network
        rng = np.random.RandomState(7919)
        t_X = T.matrix(dtype=theano.config.floatX)
        t_y = T.ivector()
        t_learning_rate = T.scalar(dtype=theano.config.floatX)

        layer1 = NNLayer(rng, t_X, 1, 256, T.nnet.relu)
#        layer2 = NNLayer(rng, layer1.output, 256, 256, T.nnet.relu)
#        layer3 = NNLayer(rng, layer2.output, 256, 256, T.nnet.relu)
        layer4 = LogisticRegression(layer1.output, 256, 4)

        cost = layer4.negative_log_likelihood(t_y)
#        params = layer4.params + layer3.params + layer2.params + layer1.params
        params = layer4.params + layer1.params
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
        # self.forward = theano.function([ t_X ], output)

    def train_once(self, learning_rate):
        self.epoch += 1
        gd_cost = self.train_model(learning_rate)
        print("{}: {}".format(self.epoch, gd_cost))
        return gd_cost

    def train_until_converge(self, target_cost):
        learning_rate = 1. / 2
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
        None
        # return self.forward(data_X)

    def test(self, data):
        None
        # y = self.predict(data.test_X)
        # print(y.reshape(1, 256))

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

if __name__ == "__main__":
    main()
