# 피마 인디언의 당뇨병 예측
# 이항 분류 문제

from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

# seed 값 생성
# seed 값을 생성하는 것의 의미?
# seed는 random 테이블 중에서 몇 번째 테이블을 불러와 쓸지 결정하는 함수이다.
# 여기서는 seed 값이 0으로 설정해 numpy와 tensorflow의 일정한 결과값을 얻기 위함이다.
# 그러나 텐서플로우 안의 내부 소프트웨어가 자체적으로 다른 랜덤 테이블을 작성하기 때문에
# 정확히 같지는 않을 수 있으므로 최종 딥러닝 결과는 여러번 실행하여 평균을 구하는 것이 가장 적절하다.
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)



# 데이터 로드
dataset = numpy.loadtxt("../dataset/pima-indians-diabetes.csv",delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]


# 모델의 설정
# 2개의 은닉층 생성
# 하나는 8개의 인풋을 받고 12개의 출력층을 갖는 relu 활성화 함수를 사용한 은닉층 생성!
# 나머지 하나는 8개의 출력층을 갖고 relu 활성화 함수를 사용한 은닉층 생성!
# 마지막은 sigmoid 함수를 사용한 최종 출력층 생성
model = Sequential()
model.add(Dense(12,input_dim=8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# 오차함수로 binary_crossentropy 
# 오차함수의 역할 : 선을 그었을때 그 가장 잘 그은 선을 찾기 위해 오차를 계산하는 함수
# 최적화를 위한 고급 경사 하강법으로 adam 사용.
# 정확도 계산
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# 모델 실행
# epochs : 전체 데이터를 한번 훑으면 epochs = 1임. 총 200번의 훑겠다는 의미
# 이때 한번에 읽어들일 양은 10개
model.fit(X,Y, epochs=200, batch_size=10)


# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1]))