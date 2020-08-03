import matplotlib.pyplot as plt
import tensorflow as tf

x_train = [[1., 2.], [2., 3.], [3., 1.], [4., 3.], [5., 3.], [6., 2]]
y_train = [[0.], [0.], [0.], [1.], [1.], [1.]]

x_test = [[5., 2.]]
y_test = [[1.]]

x1 = [x[0] for x in x_train] # 1, 2, 3, ... x_train의 첫번째 값
x2 = [x[1] for x in x_train] # 2, 3, 1, ... x_train의 두번째 값

colors = [int(y[0] % 3) for y in y_train]
plt.scatter(x1, x2, c=colors, marker='^')
plt.scatter(x_test[0][0], x_test[0][1], c="red")

plt.xlabel("x1")
plt.ylabel("x2")
#plt.show()

# x_train : 1 * 2
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# W : 2 * 1, b : 1 * 1
W = tf.Variable(tf.zeros([2, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# == sigmoid 함수
def logistic_regression(features):
    # == tf.divide(1., 1. + tf.exp(-tf.matmul(features, W) + b))
    hypothesis = tf.sigmoid(tf.matmul(features, W) + b)
    return hypothesis

# cost 함수
def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

# 판단 함수
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

def grad(features, labels):
    with tf.GradientTape() as tape:
        hypothesis = logistic_regression(features)
        loss_value = loss_fn(hypothesis, labels)
    return tape.gradient(loss_value, [W,b])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

EPOCHS = 1000
for step in range(EPOCHS+1):
    for features, labels in iter(dataset.batch(len(x_train))):
        hypothesis = logistic_regression(features)
        grads = grad(features, labels)

        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(hypothesis, labels)))

test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print("Test Result = {}".format(tf.cast(logistic_regression(x_test)>0.5, dtype=tf.int32)))
print("Testset Accuracy: {:.4f}".format(test_acc))