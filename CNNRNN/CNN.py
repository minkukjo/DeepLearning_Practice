# Convolutional Neural Network, CNN 을 사용해보자.
# CNN은 딥러닝 프레임 이미지 인식 분야에서 가장 강력한 성능을 보이는 신경망이다.
# 그 원리는 가중치 값을 곱해가며(곱하는 놈이름을 마스크라고 함) 새롭게 만들어진 컨볼루션(합성곱)을 만들어 입력 데이터로부터 더욱 정교한 특징을 추출해 낸다.
# 이러한 마스크를 여러개 만들 경우 여러개의 컨볼루션이 만들어진다.
# Conv2D() 함수를 이용해 컨볼루션 층을 추가할 수 있다.
# Conv2D의 첫번째 인자 : 마스크를 몇개 적용할지를 결정함.
# Conv2D의 두번째 인자 : kernel_size는 마스크의 크기를 결정한다. kernel_size=(행,열) 형식으로 정한다.
# Conv2D의 세번째 인자 : input_shape는 Dense층과 마찬가지로 맨 처음 층에는 입력되는 값을 알려줘야한다. input_shape=(행,열,색상 or 흑백)
# Conv2D의 마지막 인자 : activation 활성화 함수를 정의한다.
# 그 후 맥스 풀링 기법을 사용한다.
# 맥스 풀링(Max Pooling)기법이란 정해진 구역 안에서 가장 큰 값만 다음 층으로 넘기고 나머지는 버린다.
# 예를 들어 4x4 행렬을 4 구역으로 나누어 각 구역중 가장 큰 값들을 모아서 2x2행렬로 만들어주는 것이 맥스 풀링 방식이다.
# MaxPooling2D(pool_size = 2)라는 코드의 의미는 전체크기를 절반으로 줄인다는 의미이다.
# 드롭아웃과 플래튼에 대하여.
# 딥러닝 학습을 실행할때 가장 중요한 것은 과적합을 얼마나 효과적으로 피해가지는지에 달려있음.
# 간단하면서 가장 효과가 큰 방법인 Drop out기법은 은닉층에 배치된 노드중 일부를 임의로 꺼주는 것이다.
# 이렇게 랜덤하게 노드를 끔으로써 학습 데이터에 지나치게 치우쳐서 학습되는 과적합을 방지할 수 있다.
# 케라스는 이를 손쉽게 적용 가능함.
# 컨볼루션 층과 맥스풀링은 이미지를 2차원 배열인채로 다루기 때문에 이를 1차원으로 바꿔주는 Flatten 함수를 이용함.

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

import tensorflow as tf
import os
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

(X_train,Y_train), (X_test,Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32') / 255
Y_trian = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# 컨볼루션 신경망 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10)

history = model.fit(X_train,Y_trian,epochs=30,batch_size=200,validation_data=(X_test,Y_test),verbose=0,callbacks=[early_stopping_callback,checkpointer])

print("\n Test accuracy: %.4f" % (model.evaluate(X_test,Y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']
# 학습 셋의 오차
y_loss = history.history['loss']

x_len = numpy.arange(len(y_vloss))
plt.plot(x_len,y_vloss,marker='.',c="red",label='Testset_loss')
plt.plot(x_len,y_loss,marker='.',c="blue",label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
# plt.axis([0,20,0,0.35])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()