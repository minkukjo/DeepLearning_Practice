# 다중 분류 문제 해결하기

import pandas as pd
import matplotlib
import numpy
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers.core import Dense
from keras.models import Sequential
import tensorflow as tf


# seed값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('../dataset/iris.csv',names = ["sepal_length","sepal_width","petal_length","petal_width","species"])

# 그래프 표시
sns.pairplot(df, hue='species')
plt.show()

# 데이터 분류
dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]

# Y값이 문자열이기 때문에 숫자로 바꿔준다. array['iewrar','aewrewraer','aeraewre'] => array[1,2,3]로 바꾼다.
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 그런데 활성화 함수를 적용하려니 1,2,3이라서 문제가 생겼다. 그래서 다시 array[1,2,3]을 array[[1,0,0],[0,1,0],[0,0,1]]로 바꿔준다.
Y_encoded = np_utils.to_categorical(Y)

# 이렇게 Y의 값을 0과 1로만 이루어진 형태로 바꿔주는 기법을 One-hot-encoding이라고 한다.

# 소프트 맥스
# 최종 출력값이 3개중 하나여야 하므로 출력층에 해당하는 노드 수를 3개로 설정.
# 소프트 맥스를 활성화 함수로 채택
model = Sequential()
model.add(Dense(16, input_dim=4,activation='relu'))
model.add(Dense(3,activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 모델 실행
model.fit(X,Y_encoded,epochs=50,batch_size=1)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X,Y_encoded)[1]))