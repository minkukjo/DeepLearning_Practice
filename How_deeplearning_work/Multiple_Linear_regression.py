# 다중 선형 회귀(Multiple Linear regression)란?
# 공부시간만을 x(데이터)로 넣었는데 조금 더 많은 정보를 추가해 새로운 예측값을 구하기 위해
# 변수의 개수를 늘려 '다중 선형 회귀'를 만들어주어야 한다.
# y = a1x1 + a2x2 + b

import tensorflow as tf

# x1[공부한 시간],x2[과외 수업 횟수],y[성적]의 데이터 값

data =[[2,0,81], [4,4,93], [6,2,91], [8,3,97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data] # 새로 추가된 값
y_data = [y_row[2] for y_row in data]

a1 = tf.Variable(tf.random_uniform([1],0,10, dtype=tf.float64, seed = 0))
a2 = tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0)) # 새로 추가되는 값
b = tf.Variable(tf.random_uniform([1],0,100,dtype=tf.float64,seed=0))

# y = a1x1 + a2x2 + b에 대한 식
y = a1*x1 + a2*x2 + b


#최소 오차 제곱근
rmse = tf.sqrt(tf.reduce_mean(tf.square(y- y_data)))

# 학습률 값
learning_rate = 0.1

# RMSE 값을 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess:
    # sess 초기화
    sess.run(tf.global_variables_initializer())

    #2000번 학습
    for step in range(2001):
        sess.run(gradient_decent)
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a1 = %.4f, 기울기 a2 = %.4f, y 절편 b = %.4f" 
            %(step,sess.run(rmse),sess.run(a1),sess.run(a2),sess.run(b)))
