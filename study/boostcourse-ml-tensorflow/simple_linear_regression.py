import tensorflow as tf
#tf.enable_eager_execution()

# 데이터 입력
x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]

# weight와 bias 무작위로 초기화
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# learning rate 설정
learning_rate = 0.01

# epoch 설정
epoch = 100

# 학습
for i in range(epoch+1):
    # GradeintTape : tape에 context 안에서 실행된 모든 연산을 tape에 "기록"하여
    # 그래디언트를 구할 수 있게 해줌
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    # tape.gradeint 기록된 연산의 "그래디언트"를 계산
    W_grad, b_grad = tape.gradient(cost, [W,b])

    # 업데이트 : 그래디언트가 0이 되도록
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))