# 평균 제곱근 오차를 통해 여러 입력(x)가 들어오면 그에 상응하는 값을 계산하기 위해
# 임의의선을 여러개 그리고 그 오차값을 계산해서 얼마나 잘 그려졌는지 계산해야 한다.
# 이때 사용하는 것이 평균 제곱근 오차(RMSE)다.

import numpy as np

ab = [3,76]

data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    return ab[0]*x + ab[1]

def rmse(p, a):
    return np.sqrt(((p - a) ** 2).mean())

def rmse_val(predict_result,y):
    return rmse(np.array(predict_result), np.array(y))

predict_result = []

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한 시간 =%.f, 실제 점수 =%.f, 예측점수 =%.f" % (x[i],y[i],predict(x[i])))

print("rmse 최종값:" + str(rmse_val(predict_result,y)))

