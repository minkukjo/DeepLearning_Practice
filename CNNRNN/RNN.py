# 순환 신경망 (RNN)
# 문장은 여러개의 단어로 이루어져있는데 그 의미를 전달하려면 각 단어가 정해진 순서대로 입력되어야 하기 때문이다.
# 즉 여러 데이터가 순서와 관계없이 입력되던 것과 다르게, 
# 이번에는 과거의 입력된 데이터와 나중에 입력된 데이터 사이의 관계를 고려해야하는 문제가 생긴다.
# 이를 해결하기 위해 순환 신경망(Recurrent Neural Network, RNN) 방법이 고안되었다.
# 순환 신경망은 여러 개의 데이터가 순서대로 입력되었을 때 앞서 입력받은 데이터를 잠시 기억해놓는 방법이다.
# 그리고 기억된 데이터가 얼마나 중요한지를 판단해 별도의 가중치를 줘서 다음 데이터로 넘긴다.
# 모든 입력 값에 이 작업을 순서대로 실행하므로 다음 층으로 넘어가기 전 같은 층을 맴도는 것 처럼 보인다.
# 이렇게 같은 층 안에서 맴도는 성질 때문에 순환신경망이라 불린다.
# 그런데 이렇게 한 층에서 반복을 많이 해야하는 RNN의 특성상 기울기 소실문제가 더 많이 발생하고
# 이를 해결하기가 어렵다는 단점을 보완하기위해 LSTM(Long Short Term Memory) 방법을 사용한다.
# 즉 반복되기 직전에 다음 층으로 기억된 값을 넘길지 안넘길지를 관리하는 단계를 하나 더 추가하는 것이다.
# 이 LSTM을 이용해 로이터 뉴스 카테고리를 분류해보자!

from keras.datasets import reuters
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, LSTM,Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# test_split = 0.2의 의미는 20%를 테스트셋으로 사용하겠다는 의미.
# num_words = 1000의 의미는 빈도가 1~1000에 해당하는 단어만 선택해서 불러오는 것 이다.
# 기사 안의 단어중에는 거의 사용되지않는 것들이 이다. 때문에 모든 단어를 사용하면 비효율적이기 때문에
# 빈도가 높은 단어만 불러와 사용했다.
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)

# 데이터 전처리 함수 sequence를 이용해보자.
# maxlen=100은 단어 수를 100개로 맞추라는 뜻.
# 만일 입력된 기사의 단어 수가 100보다 크면 100개째 단어만 선택하고 나머진 버린다.
# 100에서 모자랄 때는 모자라는 부분을 모두 0으로 채운다.
# y데이터에 원-핫 인코딩 처리를 해서 데이터 전처리 과정을 마쳐보자.
X_train = sequence.pad_sequences(X_train,maxlen=100)
X_test = sequence.pad_sequences(X_test,maxlen=100)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# tanh? 시그모이드 함수의 크기와 위치를 조절한 함수.
# 하이퍼볼릭탄젠트의 범위는 [-1,1]까지이다. 시그모이드 함수와 달리 0을 기준으로 대칭.
# Embedding 층과 LSTM 층이 생성된 것을 확인할 수 있음.
# Embedding 
model = Sequential()
model.add(Embedding(1000,100))
model.add(LSTM(100,activation='tanh'))
model.add(Dense(46,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 모델의 실행
history = model.fit(X_train,Y_train, batch_size=100, epochs=20, validation_data=(X_test,Y_test))

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test,Y_test)[1]))
