# 베스트 모델 만들어보기
# 와인의 12가지 속성으로 화이트와인인지 레드와인인지 판별하는 딥러닝 프로그램 제작
# + K-fold 학습방법을 응용해 보겠음.
# + 모델 폴더를 만들어서 거기다 epoch 값을 넣어 보겠음
# + 그래프를 각각의 폴더에 넣어서 학습의 정확도가와 테스트셋 오차 간의 상관관계를 확인해봄
# + EarlyStopping()함수를 이용해 학습에 진전이 없으면 중단하는 함수를 추가시켜봄.

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint,EarlyStopping

import pandas as pd
import numpy
import os
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 입력
df_pre = pd.read_csv('../dataset/wine.csv',header=None)
# csv의 정보 중 랜덤으로 15%만 선별하여 사용
df = df_pre.sample(frac=0.15)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

# 5개로 쪼갬
n_fold = 5
skf = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=seed)

accuracy = []

count = 0

# 모델 설정 + K-fold

for train,test in skf.split(X,Y):
    model = Sequential()
    model.add(Dense(30, input_dim= 12 , activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    # 모델 컴파일
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # 모델 저장 폴더 설정
    MODEL_DIR = './model' + str(count) + '/'

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    
    # 모델 저장소 조건 설정
    modelpath = MODEL_DIR + '{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',verbose=1,save_best_only=True)

    # 학습 자동 중단 설정
    # patience = 100의 의미는 100번 학습할동안 val_loss값이 나아지지않으면 학습을 중단하라는 의미임.
    early_stopping_callback = EarlyStopping(monitor='val_loss',patience=100)
    
    # 모델 실행
    history = model.fit(X[train],Y[train],validation_split=0.33, epochs=3500, batch_size=500,verbose=0,callbacks=[early_stopping_callback,checkpointer])
    
    # y_vloss에 테스트셋으로 실험 결과의 오차값을 저장
    y_vloss = history.history['val_loss']

    # y_acc에 학습셋으로 측정한 정확도의 값을 저장
    y_acc = history.history['acc']

    # x 값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
    x_len = numpy.arange(len(y_acc))
    plt.plot(x_len,y_vloss,"o",c="red",markersize=3)
    plt.plot(x_len,y_acc,"o",c="blue",markersize=3)

    plt.savefig(MODEL_DIR+'result.png')
    
    # 결과 출력
    #k_accuracy =  "%.4f" % (model.evaluate(X[test],Y[test])[1])
    #accuracy.append(k_accuracy)

    count += 1

#print("\n %.f fold accuracy:" % n_fold,accuracy)

# 그래프 그려서 확인 결과 학습셋의 정확도는 시간이 흐를수록 좋아짐.
# 하지만 테스트 결과는 어느정도 이상 시간이 흐르면 더이상 나아지지않는 것을 그래프로 확인가능
# 게다가 k-fold-validation의 횟수가 3번을 넘어가면서부터 과적합으로 인해 테스트셋의 실험결과가가 점점 나빠짐.
# 때문에 학습이 진행되어도 테스트셋 오차가 줄지않으면 학습을 멈추게하는 함수가 존재함. 바로 Early Stopping
# 이 함수를 사용해 함수에 모니터할 값과 테스트 오차가 좋아지지 않아도 몇번까지 기다릴지 정함.
# 이를 early_stopping_callback에 저장함.

# 이것두 성공적~! 결과적으로 이것저것 다 더한 짬뽕 코드가 되었다.




