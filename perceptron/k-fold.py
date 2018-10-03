# K겹 교차검증
# 전체 데이터에서 70%를 학습데이터로, 30%를 테스트셋으로 했음.
# 그러나 이 정도 테스트로는 실제로 얼마나 잘 작동하는지 확신하기가 어려움
# 그래서 이를 보안하기 위해 나온 것이 K-fold cross validation이다.
# 데이터 셋을 여러개로 나누어 하나씩 테스트 셋으로 이용ㅎ고 나머지를 모두 합해서 학습셋으로 사용함.
# 5겹 교차검증의 경우 ( 1 : 학습셋 0 : 테스트 셋)
# 1 1 1 1 0 -> 결과1 
# 1 1 1 0 1 -> 결과2
# 1 1 0 1 1 -> 결과3
# 1 0 1 1 1 -> 결과4
# 0 1 1 1 1 -> 결과5
# 결과 모두 + => 최종결과

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import numpy
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('../dataset/sonar.csv',header=None)

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

# 라벨 인코더는 문자열이기 때문에 이를 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 10개의 파일로 쪼갬
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 빈 accuracy 배열
accuracy = []

# 모델의 설정, 컴파일 실행
for train,test in skf.split(X,Y):
    model = Sequential()
    model.add(Dense(24, input_dim = 60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss = 'mean_squared_error',optimizer='adam',metrics=['accuracy'])
    model.fit(X[train],Y[train], epochs=100, batch_size=5)
    k_accuracy = "%.4f" % (model.evaluate(X[test],Y[test])[1])
    accuracy.append(k_accuracy)

# 결과 출력
print("\n %.f fold accuracy:" % n_fold, accuracy)