## Boost cource DL Tensorflow - ML Basic

#### Simple Linear Regression

Linear Regression : 데이터에를 가장 잘 대변하는 직선 방정식을 찾는 것

Hypothesis : 예측한 직선 방정식

Cost : 실제 데이터와 예측한 데이터의 차이 (에러 제곱의 합의 평균)

```python
x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]

# W, b을 임의의 값으로 초기화
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# hypothesis = W * x + b
hypothesis = W * x_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))
```

Gradient descent : 이 과정을 반복하며 학습

```python
learning_rate = 0.01	# 기울기를 반영하는 정도

#Gradient descent
with tf.GradientTape() as tape:	# tape에 변수들의 변화를 기록 (W, b)
    hypothesis = W * x_data + b
    cost = tf.reduce_mean(tf.square(hypothesis-y_data))
   
W-grad, b_grad = tape.gradient(cost, [W, b])

W.assign_sub(learning_rate * W_grad)	# W = W - learning_rate * W_grad
W.assign_sub(learning_rate * b_grad)	# W = W - learning_rate * b_grad
```

