import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,LSTM

data = pd.read_csv('./data/daily_citi_bike_trip_counts_and_weather.csv',
                 parse_dates=['date'],
                 index_col=['date'],
                 usecols=['date','trips','precipitation','snowfall','max_t','min_t','dow','holiday','weekday','weekday_non_holiday']
                 )

# 展示前20行
# print(data.head(20))
# 展示后5行
# print(data.tail())

# print(data.shape)

# 按日期显示骑行数据
# plt.figure(figsize=(15,5))
# sns.lineplot(data=data, x=data.index, y=data.trips)
# plt.show()

# 按月份显示骑行数据
# data_month = data.resample('M').sum()
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

# 选取特征
cols = ['precipitation','snowfall','max_t','min_t','dow','holiday','weekday','weekday_non_holiday']

print(train_data.dtypes)
print(test_data.dtypes)


# 特征量处理
transformer = RobustScaler()
transformer = transformer.fit(train_data[cols].to_numpy())

train_data.loc[:,cols] = transformer.transform(train_data[cols].to_numpy())
test_data.loc[:,cols] = transformer.transform(test_data[cols].to_numpy())

# 目标量处理
trips_transformer = RobustScaler()
trips_transformer = trips_transformer.fit(train_data[['trips']])

train_data.loc[:,'trips'] = trips_transformer.transform(train_data[['trips']])
test_data.loc[:,'trips'] = trips_transformer.transform(test_data[['trips']])

# 将输入的时序数据 x 和标签 y 转换成适合 LSTM 模型训练的数据格式
def create_dataset(x, y, time_steps=1):
    xs, ys = [], []
    for i in range(len(x) - time_steps):
        v = x.iloc[i:(i + time_steps)].values
        xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(xs), np.array(ys)
  
time_steps = 1

x_train, y_train = create_dataset(train_data, train_data['trips'], time_steps)
x_test, y_test = create_dataset(test_data, test_data['trips'], time_steps)

# print(x_train.shape, y_train.shape)

# 定义模型
model = Sequential()
model.add(LSTM(120,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(60, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=25,batch_size=16,shuffle=True)

# 绘制训练损失和验证损失的变化曲线
plt.plot(history.history['loss'],label='train loss')
plt.plot(history.history['val_loss'],label='vall loss')
plt.legend()
plt.show()

# 在测试集上进行预测
y_pred = model.predict(x_test)
y_pred_inv = trips_transformer.inverse_transform(y_pred.reshape(1,-1))
y_test_inv = trips_transformer.inverse_transform(y_test.reshape(1,-1))

#checking the root mean sqred error
from sklearn.metrics import mean_squared_error, r2_score
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(rmse_lstm)

# 绘图
plt.figure(figsize=(15,5))
plt.plot(y_test_inv.flatten(),marker='.',label="true")
plt.plot(y_pred_inv.flatten(),marker='.',label="pred")
plt.title('LSTM')
plt.legend()
plt.show()