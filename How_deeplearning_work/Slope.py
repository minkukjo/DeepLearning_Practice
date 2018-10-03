# 코딩으로 확인하는 경사 하강법
import tensorflow as tf

data = [[2,81], [4,93],[6,91],[8,97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

# 학습률(learning rate)의 의미에 대해서
# 기울기의 부호를 바꿔 이동시킬 때 적절한 거리를 찾지 못해 너무 멀리 이동시키면
# 한점으로 수렴하는게 아니라 발산해버림.
# 딥러닝에서는 학습률을 적절히 바꿔가면서 최적의 학습률을 찾는게 중요하다.
# 케라스는 학습률을 자동으로 조절해준다.
learning_rate = 0.1

# tf로 라이브러리를 불러오고 변수의 값을 정할때는 Variable() 함수를 사용한다.
# random_uniform()은 임의의 수를 생성해 주는 함수
# [1] => 1개라는 뜻, 0,100 => 0에서 100까지 생성하라는 뜻
a = tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
b = tf.Variable(tf.random_uniform([1],0,100,dtype=tf.float64,seed=0))

y = a*x_data + b

# 평균 제곱근 오차를 다음과 같이 구현할 수 있다.
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# 텐서플로우의 GradientDescentOptimizer()를 사용해 경사하강법의 결과를 도출해냄.
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())
    # 2001번 실행 (0번째를 포함하므로)
    for step in range(2001):
        sess.run(gradient_decent)
        # gradient_decent 경사하강법이란 뜻.
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y절편 b=%.4f" %(step,sess.run(rmse),sess.run(a),sess.run(b)))

# 에포크(Epoch)는 입력 값에 대해 몇번이나 반복하여 실험했는지를 나타낸다.
# 우리가 설정한 실험을 반복하고 100번마다 결과를 내놓는다.
# 평균 제곱근 오차(RMSE)의 변화와 기울기 a가 2.3에 수렴하는 것 그리고 y절편인 b가 79에 수렴하는 과정을 볼 수 있다.
# 이렇게 하면 최소 제곱법을 안쓰고 평균 제곱근 오차를 구하고