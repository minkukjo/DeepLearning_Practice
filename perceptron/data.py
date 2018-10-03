import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../dataset/pima-indians-diabetes.csv', 
                names=["pregnant","plasma","pressure","thickness","insulin","BMI","predigree","age","class"])

print(df.head(5))
# 출력 시 0부터 나오는데 파이썬에서는 숫자를 0부터 세기때문에 0~4의 숫자가 나온다.

print(df.info()) 
# 각 Column별 수와 정보의 형식

print(df.describe()) 
# 정보별 특징을 자세히 보는 함수

print(df[['pregnant','class']])
# 데이터 중 일부 컬럼만 보고 싶을때 사용하는 경우


print(df[['pregnant','class']].groupby(['pregnant'], as_index = False).mean().sort_values(by='pregnant',ascending=True))
# 데이터의 특정 항목과 클래스간의 상관관계를 확률로 계산해 오름차순으로 정리.

################################################################################################

plt.figure(figsize=(12,12))

sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white',annot=True)

plt.show()

# 그래프로 상관관계를 만들고 class와 가장 상관관계가 높은 항목을 한눈에 볼 수 있음.

################################################################################################

grid = sns.FacetGrid(df,col='class')
grid.map(plt.hist,'plasma',bins=10)
plt.show()

# 막대그래프를 이용해 각 클래스가 0인 경우의 상관관계와 1인 경우의 상관관계를 보여줌.
# 데이터 전처리 과정은 딥러닝을 비롯한 모든 머신러닝의 성능 향상에 중요한 역할을 한다.

################################################################################################