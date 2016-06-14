import numpy as np
import theano
from theano import tensor as T

class MyData(object):
    def load(self):
        # Load from a data file
        data_X_y = np.loadtxt('ex1data2.txt', dtype=theano.config.floatX, delimiter=',')
        # Split X, y
        columns = data_X_y.shape[1]
        data_X = data_X_y[:, 0:columns - 1]
        data_y = data_X_y[:, columns - 1:columns]
        self.n_features = columns - 1
        self.n_data = len(data_y)
        # Insert ones for bias
        data_bias = np.ones((self.n_data, 1), dtype=theano.config.floatX)
        data_X = np.c_[ data_X, data_bias ]
        # Normalization
        mu = [ None ] * columns
        sigma = [ None ] * columns
        for i in range(columns - 1):
            mu[i] = np.mean(data_X[:, i:i + 1])
            sigma[i] = np.std(data_X[:, i: i + 1])
            data_X[:, i:i + 1] -= mu[i]
            data_X[:, i:i + 1] /= sigma[i]
        # Make shared
        print("  Shape of X: {}".format(data_X.shape))
        print("  Shape of y: {}".format(data_y.shape))
        self.data_X = theano.shared(data_X, borrow=True)
        self.data_y = theano.shared(data_y, borrow=True)
        # save information
        self.data_np_X = data_X
        self.data_np_y = data_y
        self.mean = mu
        self.mu = sigma

class LinearRegression(object):
    def __init__(self, t_X, n_features):
        # Weights and bias
        Wb = np.zeros((n_features + 1, 1), dtype=theano.config.floatX)
        self.Wb = theano.shared(Wb, borrow=True)

        # prediction
        # t_X (m, features) * Wb (features) => (m, 1)
        self.output = T.dot(t_X, self.Wb)

    def sqr_errors(self, t_y):
        return T.sum((self.output - t_y) ** 2)

class MyLearn(object):
    def __init__(self):
        self.epoch = 0

    def build_model(self, data):
        # tensor: learning rate, X, y
        t_lr = T.scalar(dtype=theano.config.floatX)
        t_X = T.matrix(dtype=theano.config.floatX)
        t_y = T.matrix(dtype=theano.config.floatX)

        # training model
        regress = LinearRegression(t_X, data.n_features)
        cost = regress.sqr_errors(t_y)

        grads = T.grad(cost, wrt=regress.Wb)
        updates = [ (regress.Wb, regress.Wb - t_lr * grads) ]
        givens = { t_X: data.data_X, t_y: data.data_y }
        self.train_model = theano.function([ t_lr ], cost, updates=updates, givens=givens)

        # forward model
        self.forward_model = theano.function([ t_X ], regress.output)

    def forward(self, data_X):
        return self.forward_model(data_X)

    def train_once(self, lr=0.01):
        return self.train_model(lr)

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

def main():
    data = MyData()
    data.load()
    print("Loaded {} samples, {} features".format(data.n_data, data.n_features))

    mml = MyLearn()
    mml.build_model(data)

    mml.train_until_converge(0.1)

    y = mml.forward(data.data_np_X)
    print("Price : Predicted : Difference")
    for i in range(data.n_data):
        price = data.data_np_y[i][0]
        pred = y[i][0]
        diff = pred - price
        print("{0:.0f} : {1:.0f} : {2:.0f}".format(price, pred, diff))

if __name__ == "__main__":
    main()
