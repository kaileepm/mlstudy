import tensorflow as tf
import numpy as np

coverge_threshold = 0.0001

x = tf.placeholder("float");
y = tf.placeholder("float");
theta = tf.Variable([.0, .0], name="theta")
y_model = theta[0] + tf.mul(x, theta[1])
cost = tf.square(y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

model = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(model)

	theta_old_0 = 0;
	theta_old_1 = 0;

	while True:
		for i in range(100):
			x_train = np.random.rand()
			y_train = x_train * 17 + 23
			session.run(train_op, feed_dict={x: x_train, y: y_train})

		theta_new = session.run(theta)
		theta_new_0 = theta_new[0];
		theta_new_1 = theta_new[1];

		print("Predicted model: {0:.3f} + {1:.3f} * x1".format(theta_new_0, theta_new_1))

		diff = abs(theta_new_0 - theta_old_0) + abs(theta_new_1 - theta_old_1)
		if diff < coverge_threshold:
			break

		theta_old_0 = theta_new_0
		theta_old_1 = theta_new_1
