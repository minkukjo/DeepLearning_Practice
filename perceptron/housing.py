# 선형회귀 적용하기
# 집값 예측해보기
# 깨끗한 공기와 집값의 상관관계를 선형회귀를 이용해 집값을 예측해보자

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
import tensorflow as tf

# seed값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("../dataset/housing.csv",delim_whitespace=True,header=None)

dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=seed)

model = Sequential()
model.add(Dense(30,input_dim=13, activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(X_train,Y_train,epochs=200,batch_size=10)

# 예측 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()

for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제 가격 : {:.3f}, 예상가격 {:.3f}".format(label,prediction))
