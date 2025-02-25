# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from keras.api.models import load_model

sns.set_style("darkgrid")

# 引入数据集
data = pd.read_csv('./data/daily_citi_bike_trip_counts_and_weather.csv',
                   parse_dates=['date'],
                   index_col=['date'],
                   usecols=['date',
                            'trips',
                            'precipitation',
                            'snowfall',
                            'max_t',
                            'min_t',
                            'average_wind_speed',
                            'dow',
                            'holiday',
                            # 'stations_in_service',
                            'weekday',
                            'weekday_non_holiday',
                            'month',
                            'dt',
                            'day',
                            'year'])

train_percentage = 0.8
train_size = int(len(data) * train_percentage)
test_size = len(data) - train_size
train_data,test_data = data.iloc[0:train_size],data.iloc[train_size:len(data)]

# 选取特征
cols = ['precipitation',
        'snowfall',
        'max_t',
        'min_t',
        'average_wind_speed',
        'dow',
        'holiday',
        # 'stations_in_service',
        'weekday',
        'weekday_non_holiday',
        'dt',
        'day',
        'year']

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
    
model_path = './best_model/2/final_bike_usage_model.keras'

model = load_model(model_path)  # 加载模型

# 在测试集上进行预测
y_pred = model.predict(x_test)
y_pred_inv = trips_transformer.inverse_transform(y_pred.reshape(1,-1))
y_test_inv = trips_transformer.inverse_transform(y_test.reshape(1,-1))

# 均方根误差
from sklearn.metrics import mean_squared_error, r2_score
rmse_lstm = round(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)),2)
print(rmse_lstm)

# 绘图
plt.figure(figsize=(12,4))
plt.plot(y_test_inv.flatten(),marker='.',label="true")
plt.plot(y_pred_inv.flatten(),marker='.',label="pred")
plt.title('LSTM')
plt.legend()
plt.show()