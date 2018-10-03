# 손글씨 이미지 인식
# 케라스의 mnist를 이용해 손글씨 데이터를 불러오고 이를 토대로 컴퓨터가 어떻게 숫자를 판별하는지 확인

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
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

# MNIST의 손글씨 데이터는 총 70000개의 이미지중 6만개를 학습용, 1만개를 테스트용으로 미리 구분해놓고 있음.
(X_train,Y_class_train), (X_test,Y_class_test) = mnist.load_data()

# print("학습셋 이미지 수 :%d 개" % (X_train.shape[0]))
# print("테스트셋 이미지 수 :%d 개" % (X_test.shape[0]))

# 학습 이미지중 하나를 흑백으로 출력해봤음
# plt.imshow(X_train[0],cmap='Greys')
# plt.show()

# 5라는 형태로 나오는것을 확인 가능
# for x in X_train[0]:
#     for i in x:
#         sys.stdout.write('%3d ' % i)
#     sys.stdout.write('\n')

# 이러한 28x28 2차원 배열을 784개의 1차원 배열로 바꿔주어야 한다.
# 이를 위해 reshape()함수를 이용한다.

X_train = X_train.reshape(X_train.shape[0],784)

# 케라스는 데이터를 0에서 1사이의 값으로 변환한 다음 구동할때 최적의 성능을 보인다.
# 따라서 현재 0~255 사이의 값으로 이루어진 값을 0~1사이의 값으로 바꿔야 한다.
# 바꾸는 방법은 각 값을 255로 나누는 것이다. 
# 이러한 데이터의 폭이 클때 적절한 값으로 분산의 정도를 바꾸는 과정을 데이터 정규화(nomalization)이라고 한다.
# 현재 주어진 값은 0~255까지의 정수이므로 정규화를 위해 255로 나누어주려면 이 값을 실수형으로 바꿔야 한다.
X_train = X_train.astype('float64')
X_train = X_train / 255

# print("class : %d" % (Y_class_train[0]))

# test에도 똑같은 작업을 함.
X_test = X_test.reshape(X_test.shape[0],784).astype('float64') / 255

# 그런데 여기서 아리리스 품종을 예측할때 우리는 원-핫 인코딩 방식을 적용해야한다고 배웠었다.
# 즉 0~9까지의 정수형 값을 갖는 현재 형태에서 0과 1로 이루어진 벡터로 값을 수정해야 한다.
# 예를들어 class가 '3'이라면 [0,0,1,0,0,0,0,0,0,0] 이런식으로 바꿔줘야한다
# np_utils.to_categorical() 함수가 그 함수이며 to_categorical(클래스(정답),클래스(정답)의 개수) 두개의 인자를 받는다.

# 벡터 normalizate
Y_trian = np_utils.to_categorical(Y_class_train,10)
Y_test = np_utils.to_categorical(Y_class_test,10)

model = Sequential()
model.add(Dense(512,input_dim=784,activation='relu'))
# softmax를 사용하는 이유?
# 입력받은 값을 토대로 0~1사이의 값으로 모두 정규화하며 총합은 항상 1이되는 특성을 가진 함수이다.
# 만약 이 프로그램의 경우 0~9사의 값이 있고 입력이 5인경우 5번째 배열의 값이 큰 형태로 나오고 나머지 값은 낮게 나올 것임을 알 수 있다.
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10)

# 모델의 실행


history = model.fit(X_train,Y_trian,epochs=30,batch_size=200,validation_data=(X_test,Y_test),verbose=0,callbacks=[early_stopping_callback,checkpointer])

print("\n Test accuracy: %.4f" % (model.evaluate(X_test,Y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']
# 학습 셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
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