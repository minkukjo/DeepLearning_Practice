# 폐암 수술 환자의 생존율 예측하기

#모듈 가져오기
from keras.models import Sequential
from keras.layers import Dense

import numpy
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 준비된 수술 환자 데이터를 불러입니다.
Data_set = numpy.loadtxt("./dataset/ThoraricSurgery.csv",delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장합니다.
X = Data_set[:,0:17]
Y = Data_set[:,17]

# 딥러닝 구조를 결정한다. (모델을 결정하고 실행한다.)
model = Sequential()
model.add(Dense(30, input_dim = 17, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
# activation에 대하여
# 다음 층으로 어떻게 값을 넘길지 결정하는 부분입니다.
# 여기서는 가장 많이 사용되는 relu와 sigmoid 함수를 사용하게끔 지정하고 있다.

# 딥러닝을 실행한다.
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
# loss, optimizer의 의미에 대하여
# loss : 한번 실행될때마다 오차 값을 추적하는 함수
# optimizer : 오차를 어떻게 줄여 나갈지 정하는 함수 입니다.
model.fit(X,Y,epochs=30,batch_size=10)

# 결과를 출력합니다.

loss = (model.evaluate(X,Y)[0])
score = (model.evaluate(X,Y)[1])

print("\n Loss : %.4f" % loss)

print("\n Accuracy : %.4f" % score)

