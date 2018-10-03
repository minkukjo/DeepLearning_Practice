# 과적합 피하기
# 광석과 일반 돌에 음파 탐지기를 쏜 후 그 결과를 데이터로 저장
# 오차 역전파 알고리즘이 얼마나 광석과 돌을 구분하는데 효과적인지 검증

from keras.models import Sequential,load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 입력
df = pd.read_csv('../dataset/sonar.csv',header=None)
# print(df.info())
# print(df.head())

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

# 문자열 반환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# + 학습셋과 테스트셋의 구분
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state = seed)


# 모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss = 'mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
# 모델 실행
model.fit(X_train,Y_train,epochs=200,batch_size=5)

# 테스트를 위해 모델 저장
model.save('my_model.h5')

# 메모리의 모델 삭제
del model 

# 모델 불러옴
model = load_model('my_model.h5')

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X_test,Y_test)[1]))


# 그렇다면 과적합이란 무엇인가?
# over fitting이란 학습 데이터 셋 안에서는 일정 수준 이상의 예측 정확도를 보이지만
# 새로운 데이터에 적용하면 잘 맞지 않는 것을 의미한다.
# 과적합의 원인으로는 층이 너무 많거나, 변수가 복잡해서 발생하기도 하고
# 테스트셋과 학습셋이 중복되는 경우에 생기기도 한다.
# 딥러닝은 학습단계에서 입력,은닉층,출력층 노드들에 상당히 많은 변수들이 투입된다.
# 따라서 딥러닝을 진행하는 동안 과적합에 빠지지 않게 늘 주의해야 한다.

# 과적합 방지법
# 학습을 하는 데이터셋(data)과 이를 테스트할 데이터셋(label) 완전히 구분
