#importing the necessary libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from pylab import rcParams
# import tensorflow as tf
# from tensorflow import keras
sns.set_style("darkgrid")

data = pd.read_csv('./data/daily_citi_bike_trip_counts_and_weather.csv',
                 parse_dates=['date'],
                 index_col=['date'],
                 usecols=['date','trips','dow','holiday','weekday','weekday_non_holiday','snowfall','min_temperature','max_temperature','precipitation']
                 )

# print(df.head(20))
# print(df.tail())
# print(df.shape)

data_month = data.resample('M').sum()

# 按日期显示骑行数据
plt.figure(figsize=(15,5))
sns.lineplot(data=data, x=data.index, y=data.trips)
plt.show()

# 按月份显示骑行数据
# plt.figure(figsize=(15,5))
# sns.lineplot(data=data_month, x=data_month.index, y=data_month.trips)
# plt.show()

# 按星期和是否假期显示骑行数据
# plt.figure(figsize=(15,5))
# sns.pointplot(x='dow',y='trips',hue='holiday',data=data)
# plt.show()

# 分割用于训练的数据和用于测试的数据
# 90% 用于训练，10% 用于测试
train_size = int(len(data) * 0.9)
test_size = len(data) - train_size
train_data,test_data = data.iloc[0:train_size],data.iloc[train_size:len(data)]
print(len(train_data), len(test_data))