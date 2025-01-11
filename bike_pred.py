# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.api.models import Sequential
from keras.api.layers import Dense,Flatten,Dropout,LSTM

# 引入数据集
data = pd.read_csv('./data/daily_citi_bike_trip_counts_and_weather.csv',
                   parse_dates=['date'],
                   index_col=['date'],
                   usecols=['date','trips','precipitation','snowfall','max_t','min_t','dow','holiday','weekday','weekday_non_holiday'])

# 展示前20行
print(data.head(20))