import pandas as pd
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn
from scipy import stats

plt.style.use('ggplot')

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({'font.size':5})
train = pd.read_csv("data/train.csv",parse_dates=["datetime"])

# 기온에 관한 구체적인 데이터
# print(train.temp.describe())

# null 인 데이터가 있는지
# print(train.isnull().sum())

train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second

# 바차트로 데이터 분석 해보기
# figure, ( (ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3)
# figure.set_size_inches(18,8)

# seaborn.barplot(data=train,x="year",y="count",ax=ax1)
# seaborn.barplot(data=train,x="month",y="count",ax=ax2)
# seaborn.barplot(data=train,x="day",y="count",ax=ax3)
# seaborn.barplot(data=train,x="hour",y="count",ax=ax4)
# seaborn.barplot(data=train,x="minute",y="count",ax=ax5)
# seaborn.barplot(data=train,x="second",y="count",ax=ax6)

# ax1.set(ylabel='Count',title="year rent count")
# ax2.set(xlabel='month',title="month rent count")
# ax3.set(xlabel='day',title="day rent count")
# ax4.set(xlabel='hour',title="time rent count")


# plt.title("Plot")
# plt.show()


# 박스형태로 데이터 분석 해보기
# fig,axes = plt.subplots(nrows=2,ncols=2)
# fig.set_size_inches(12,10)
# seaborn.boxplot(data=train,y="count",orient="v",ax=axes[0][0])
# seaborn.boxplot(data=train,y="count",x="season",orient="v",ax=axes[0][1])
# seaborn.boxplot(data=train,y="count",x="hour",orient="v",ax=axes[1][0])
# seaborn.boxplot(data=train,y="count",x="workingday",orient="v",ax=axes[1][1])

# axes[0][0].set(ylabel='Count',title="rent count")
# axes[0][1].set(xlabel='Season',ylabel='Count',title="Season rent count")
# axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Hour Of the Day rent count")
# axes[1][1].set(xlabel='Working Day',ylabel='Count',title= "Working Day rent count")

# plt.show()

train["dayofweek"] = train["datetime"].dt.dayofweek
# print(train["dayofweek"].value_counts())

# 포인트 그래프 형태로 데이터 분석 해보기
# ax1,2,3로 쪼개면 한번에 보기는 편한데 너무 작아서 보기가 힘듬.
# 차라리 그냥 하나씩 보는게 낫더라.
# 그래도 여러개 그래프를 한눈에 봐야하는 상황이라면 이렇게 써야함
# fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5)
# fig.set_size_inches(100,100)

# seaborn.pointplot(data=train,x="hour",y="count",ax=ax1)
# seaborn.pointplot(data=train,x="hour",y="count",hue="workingday",ax=ax2)
# seaborn.pointplot(data=train,x="hour",y="count",hue="dayofweek",ax=ax3)
# seaborn.pointplot(data=train,x="hour",y="count",hue="weather",ax=ax4)
# seaborn.pointplot(data=train,x="hour",y="count",hue="season",ax=ax5)

# plt.show()

# heatmap을 통해 데이터간의 상관관계를 알아본다.

# corrMatt = train[ ["temp","atemp","casual","registered","humidity","windspeed","count"]]
# corrMatt = corrMatt.corr()
# print(corrMatt)

# mask = numpy.array(corrMatt)
# mask[numpy.tril_indices_from(mask)] = False

# fig,ax = plt.subplots()
# fig.set_size_inches(20,10)
# seaborn.heatmap(corrMatt,mask=mask,vmax=8,square=True,annot=True)

# plt.show()

# regplot을 사용해 점으로 산점도를 확인
# 산점도를 확인해보면 풍속이 0에 많은걸로 보아 아무래도 관측되지않은 수치는 0으로 기록한게 아닐까 추측가능

# fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
# fig.set_size_inches(12,5)
# seaborn.regplot(x="temp",y="count",data=train,ax=ax1)
# seaborn.regplot(x="windspeed",y="count",data=train,ax=ax2)
# seaborn.regplot(x="humidity",y="count",data=train,ax=ax3)

# plt.show()

def concatenate_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)

train["year_month"] = train["datetime"].apply(concatenate_year_month)
print(train.shape)

fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(18,4)

seaborn.barplot(data=train,x="year",y="count",ax=ax1)
seaborn.barplot(data=train,x="month",y="count",ax=ax2)

fig,ax3 = plt.subplots(nrows=1,ncols=1)
fig.set_size_inches(100,4)

seaborn.barplot(data=train,x="year_month",y="count",ax=ax3)
plt.show()
