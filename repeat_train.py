# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from keras.api.callbacks import ModelCheckpoint
from keras.api.models import Sequential,load_model
from keras.api.layers import Dense,Flatten,Dropout,LSTM

sns.set_style("darkgrid")

# 检查 TensorFlow 是否能够检测到 GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("GPU is not available.")
    
data_name = 'daily_citi_bike_trip_counts_and_weather'

# 引入数据集
data = pd.read_csv('./data/' + data_name + '.csv',
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

# 基于当前时间创建路径 作为基础路径使用
base_path = "./model/{0:%Y-%m-%d %H-%M-%S}/".format(datetime.now())

# 临时路径 用于保存训练过程中回调函数保存的模型
temp_path = './model/temp/temp_bike_pred_model.keras'

# 设定程序一次训练的模型数量
repeat = 40
for i in range(repeat):
    
    # 引入数据指标记录表
    train_df = pd.read_excel('train.xlsx')

    # 记录开始时间
    start_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())

    # 设置保存模型的路径
    model_save_path = base_path + str(i) +'/bike_pred_model.keras'

    # 创建保存模型的文件夹（如果没有的话）
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 1. 在训练时使用 ModelCheckpoint 回调保存最佳模型
    checkpoint = ModelCheckpoint(
        temp_path,                # 模型保存路径
        monitor='val_loss',             # 监视的指标，这里监视验证集损失
        save_best_only=True,            # 仅在验证损失最小时保存
        save_weights_only=False,        # 保存整个模型（包括模型架构、权重、优化器状态）
        verbose=1                       # 输出信息
    )

    # 2. 定义模型
    l1 = 144
    d1 = 0.4
    l2 = 80
    d2 = 0.3
    model = Sequential()
    model.add(LSTM(l1,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Dropout(d1))

    model.add(LSTM(l2, activation='relu'))
    model.add(Dropout(d2))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # 3. 训练模型 加入checkpoint回调
    epochs = 1500
    batch_size = 128
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[checkpoint]  # 加入 checkpoint 回调
    )

    # 记录结束时间
    end_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    
    min_val_loss = round(min(history.history['val_loss']),6)
    # print(f"验证损失的最小值: {min_val_loss}")

    # 绘制训练损失和验证损失的变化曲线
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'],label='train loss')
    plt.plot(history.history['val_loss'],label='vall loss')
    plt.legend()
    # plt.show()
    plt.savefig(base_path + str(i) +'/loss ' + str(min_val_loss) + '.png')

    model = load_model(temp_path)  # 加载模型

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
    # plt.show()
    plt.savefig(base_path + str(i) +'/rmse '+ str(rmse_lstm) +'LSTM.png')
    
    # 训练完成后保存最终模型（保存整个模型，包含架构和权重）
    model.save(base_path + str(i) +'/final_bike_usage_model.keras')

    # 记录数据指标
    new_df = pd.DataFrame([[start_time,end_time,data_name,i,train_percentage,time_steps,l1,d1,l2,d2,epochs,batch_size,rmse_lstm,min_val_loss]],columns=['start_time','end_time','data_name','index','train_percentage','time_steps','l1','d1','l2','d2','epochs','batch_size','rmse_lstm','min_val_loss'])
    save_data = train_df._append(new_df)
    save_data.to_excel('train.xlsx',index=False)
    print('数据记录完成')