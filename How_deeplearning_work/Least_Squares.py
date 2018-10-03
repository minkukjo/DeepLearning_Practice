import numpy as np

x = [2,4,6,8]
y = [81,93,91,97]

mx = np.mean(x)
my = np.mean(y)

print("x의 평균값:",mx)
print("y의 평균값:",my)

divisor = sum([(i - mx)**2 for i in x])

def top(x, mx, y , my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d

dividend = top(x,mx,y,my)

print("분자:",dividend)
print("분모:",divisor)


a = dividend / divisor

b = my - (mx*a)

print("기울기 a =", a)
print("y 절편 b=",b)

# 최소 제곱법이란 집합 x (공부시간, 데이터)와 집합 y(성적, 레이블)이 존재할때 이 x와 y의 집합을 가장 잘 표현하는 기울기 a와 y절편 b를 정확히 찾아낼때 사용하는 방법이다.
# 최소 제곱법(method of least squares)을 이용하면 데이터와의 오차가 가장 작은 기울기와 y절편을 바로 구할 수 있다. 
# 회귀 분석에서 사용하는 표준 방식이며, 실험이나 관찰을 통해 얻은 데이터를 분석하여 미지의 상수를 구할 때 사용하는 공식이다.
# 이 최소 제곱법으로 주어진 데이터 집합에 오차가 가작 적은 직선을 그려주는 기법이며 이는 주로 회귀분석에서 사용하는 표준 방식이라고 할 수 있다.
