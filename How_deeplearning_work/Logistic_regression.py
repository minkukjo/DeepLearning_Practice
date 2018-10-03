# 로지스틱 회귀란?
# 참과 거짓중에 하나를 내놓는 과정이다.
# 참인지 거짓인지를 구분하는 로지스틱 회귀의 원리를 이용해 '참, 거짓 판단 장치'를 만들어 주어진 입력 값의 특징을 추출합니다.
# 이를 저장해 model을 만들고 비슷한 질문이 오면 지금까지 만들어진 model을 꺼내어 답을 함.
# 이것이 딥러닝의 동작원리이다.

# 선형회귀에서는 공부한 시간과 성적사이의 관계를 좌표로 나타냈을때,
# 좌표의 형태가 직선으로 해결되는 선형 회귀를 사용하기에 적절했다.
# 그러나 직선으로 해결하기엔 적절하지 않은 형태도 존재한다.
# 로직스틱 회귀는 선형회귀와 마찬가지로 적절한 선을 그려가는 과정이다.
# 그러나 차이점이 있다면 직선이 아니라 참과 거짓사이를 구분하는 S자 형태의 선을 그어주는 작업이다.
# 이러한 S자 형태로 그래프가 그려지는 함수가 바로 시그모이드 함수이다.
# 시그모이드 식은 y = 1/ 1+e^(ax+b)의 형태를 갖는다.
# a는 그래프의 경사도를 의미한다. b는 그래프의 좌우 이동을 의미한다.
# a는 값이 작아지면 오차가 무한대로 커진다. 그러나 a 값이 커진다고 해서 오차값이 무한대로 커지진 않는다.
# b는 2차함수 형태이다.
# 우리는 y = 0.5에 대칭하는 두개의 로그함수를 그린다.
# 하나는 y가 0일때 오차가 무한대에 수렴하는 그래프이고
# 다른 하나는 y가 1일때 오차가 무한대에 수렴하는 그래프이다.

import tensorflow as tf
import numpy as np

data = [[2,0], [4,0], [6,0], [8,1], [10,1], [12,1], [14,1]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

a = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed = 0))
b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed = 0))

#y 시그모이드 함수의 방정식을 세운다.
y = 1/(1 + np.e**(a * x_data + b))

# loss를 구하는 함수
loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1-y))

# 학습률 값
learning_rate = 0.5

# loss를 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(60001):
        sess.run(gradient_decent)
        if i % 6000 == 0:
            print("Epoch: %.f, loss = %.4f, 기울기 a = %.4f, y 절편 = %.4f" % (i, sess.run(loss), sess.run(a), sess.run(b)))
