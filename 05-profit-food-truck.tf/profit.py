import tensorflow as tf
import numpy as np

# load data as two dimentional array
data = np.loadtxt(open("ex1data1.txt"), delimiter=',')
data_x1 = data[:, 0]
data_y = data[:, 1]

# placeholder for training data
x = tf.placeholder("float");
y = tf.placeholder("float");
theta = tf.Variable([0.0, 0.0], name="theta")
y_model = theta[0] + tf.mul(x, theta[1])
# cost = tf.square(y - y_model)
cost = tf.reduce_mean(tf.square(y_model - y))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

model = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(model)
	for loop in range(100):
		for i in range(len(data_y)):
#			x_train = np.random.rand()
#			y_train = x_train *10 + 5
			x_train = data_x1[i]
			y_train = data_y[i]
			session.run(train_op, feed_dict={x: x_train, y: y_train})
#			session.run(train_op, feed_dict={x: data_x1[i], y: data_y[i]})

		theta_value = session.run(theta)
		print("Predicted model: {0:.3f} + {1:.3f} * x1".format(theta_value[0], theta_value[1]))

