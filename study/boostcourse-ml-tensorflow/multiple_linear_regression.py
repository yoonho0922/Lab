import tensorflow as tf
import numpy as np

# 데이터 입력
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

# weight와 bias 무작위로 초기화
W = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1]))

# learning rate 설정
learning_rate = 0.000001

# 계산한 n * 1 행렬 반환
def predict(X):
    return tf.matmul(X, W) + b

n_epochs = 10000
for i in range(n_epochs+1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X)-y)))

    # W_grad : 3 * 1 행렬
    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 1000 == 0:
        print("{:5}|{:10.4f}|".format(i, cost.numpy()))
