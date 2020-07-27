import tensorflow as tf
import numpy as np

data = np.array([
    #X1,  X2,  X3,  y
    [73., 80., 75., 152.],
    [93., 88., 93., 185.],
    [89., 91., 90., 180.],
    [96., 98., 100., 196.],
    [73., 66., 70., 142.]
], dtype=np.float32)

X = data[:, :-1]
y = data[:, [-1]]

#initialize W, b
W = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001

def predict(X):
    return tf.matmul(X, W) + b

n_epochs = 1000
for i in range(n_epochs+1):
    # record the gradient of the cost function
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X)-y)))

    # calculates the gradient of the loss
    W_grad, b_grad = tape.gradient(cost, [W, b])

    # updates parameters (W and b)
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5}|{:10.4f}|".format(i, cost.numpy()))
