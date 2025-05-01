# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna  # 添加 optuna 库

sns.set_style("darkgrid")

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据集名称
data_name = 'daily_citi_bike_trip_counts_and_weather'

# 引入数据集
data = pd.read_csv('./data/' + data_name + '.csv',
                   parse_dates=['date'],
                   index_col=['date'])

# 划分训练集、验证集和测试集
train_percentage = 0.7
val_percentage = 0.8
test_percentage = 0.9
train_size = int(len(data) * train_percentage)
val_size = int(len(data) * val_percentage)
test_size = int(len(data) * test_percentage)
train_data = data.iloc[0:train_size]
val_data = data.iloc[train_size:val_size]
test_data = data.iloc[val_size:test_size]

# 选取特征
cols = [
  'precipitation',
  'snow_depth',
  'snowfall',
  'max_t',
  'min_t',
  'average_wind_speed',
  # 'dow',
  # 'year',
  'month',
  'stations_in_service',  # 取消注释，添加该特征
  'weekday',
  'weekday_non_holiday',
  'dt',
  'season'
  ]

# 特征量标准化
transformer = RobustScaler()
transformer = transformer.fit(train_data[cols].to_numpy())

train_data.loc[:, cols] = transformer.transform(train_data[cols].to_numpy())
val_data.loc[:, cols] = transformer.transform(val_data[cols].to_numpy())
test_data.loc[:, cols] = transformer.transform(test_data[cols].to_numpy())

# 目标量标准化
trips_transformer = RobustScaler()
trips_transformer = trips_transformer.fit(train_data[['trips']])

train_data.loc[:, 'trips'] = trips_transformer.transform(train_data[['trips']])
val_data.loc[:, 'trips'] = trips_transformer.transform(val_data[['trips']])
test_data.loc[:, 'trips'] = trips_transformer.transform(test_data[['trips']])


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
x_val, y_val = create_dataset(val_data, val_data['trips'], time_steps)
x_test, y_test = create_dataset(test_data, test_data['trips'], time_steps)

# 转换为 PyTorch Tensor
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# 定义 input_size
input_size = x_train.shape[2]

# 定义损失函数
criterion = nn.HuberLoss()

# 定义批量大小
batch_size = 64

# 基于当前时间创建路径 作为基础路径使用
base_path = "./model/{0:%Y-%m-%d %H-%M-%S}/".format(datetime.now())

# 临时路径 用于保存训练过程中回调函数保存的模型
temp_path = './model/temp/temp_bike_pred_model.pth'

# 引入数据指标记录表
train_df = pd.read_excel('train.xlsx')

# 设置保存模型的路径
model_save_path = base_path + '/bike_pred_model.pth'

# 创建保存模型的文件夹（如果没有的话）
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 记录开始时间
start_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())

# 自定义LSTM模型
# 双向LSTM + CNN
class MYLSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
    super(MYLSTMModel, self).__init__()
    self.conv1 = nn.Conv1d(input_size, 48, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(48, 64, kernel_size=3, padding=1)
    self.lstm1 = nn.LSTM(64, hidden_size1, batch_first=True, bidirectional=True)
    self.dropout1 = nn.Dropout(dropout1)
    self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
    self.dropout2 = nn.Dropout(dropout2)
    self.fc = nn.Linear(hidden_size2 * 2, 1)
    
  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = x.permute(0, 2, 1)
    
    out, _ = self.lstm1(x)
    out = self.dropout1(out)
    out, _ = self.lstm2(out)
    out = self.dropout2(out)
    
    out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
    return out

# 定义目标函数，用于 optuna 调优
def objective(trial):
    # 定义超参数搜索空间
    hidden_size1 = trial.suggest_int('hidden_size1', 64, 256, step=16)
    hidden_size2 = trial.suggest_int('hidden_size2', 32, 128, step=16)
    dropout1 = trial.suggest_float('dropout1', 0.1, 0.5, step=0.05)
    dropout2 = trial.suggest_float('dropout2', 0.1, 0.5, step=0.05)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    
    # 创建模型
    model = MYLSTMModel(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 训练模型
    final_val_loss = float('inf')
    for epoch in range(800):  # 限制训练轮数以加速调优
      model.train()
      optimizer.zero_grad()
      outputs = model(x_train)
      loss = criterion(outputs.squeeze(), y_train)
      loss.backward()
      optimizer.step()
      
      # 验证集上的损失
      model.eval()
      with torch.no_grad():
          val_outputs = model(x_val)
          val_loss = criterion(val_outputs.squeeze(), y_val)
      
      # 更新最终验证损失
      final_val_loss = val_loss.item()
    
    return final_val_loss

# 使用 optuna 进行超参数调优
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)  # 设置调优的试验次数

# # 获取最佳超参数
# best_params = study.best_params
# hidden_size1 = best_params['hidden_size1']
# hidden_size2 = best_params['hidden_size2']
# dropout1 = best_params['dropout1']
# dropout2 = best_params['dropout2']
# learning_rate = best_params['learning_rate']

# 手动设置参数
hidden_size1 = 128
hidden_size2 = 80
dropout1 = 0.4
dropout2 = 0.3
learning_rate = 0.0001

# 输出最佳超参数
# print("Best hyperparameters:", best_params)

# 使用最佳超参数创建最终模型
model = MYLSTMModel(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# 训练模型
train_losses, val_losses = [], []
best_val_loss = float('inf')
best_epoch = 0

epochs = 4000  # 设置训练轮数

# 训练模型
for epoch in range(epochs):
  model.train()
  optimizer.zero_grad()

  # 前向传播
  outputs = model(x_train)
  loss = criterion(outputs.squeeze(), y_train)

  # 反向传播和优化
  loss.backward()
  optimizer.step()

  # 验证集上的损失
  model.eval()
  with torch.no_grad():
    val_outputs = model(x_val)
    val_loss = criterion(val_outputs.squeeze(), y_val)

  train_losses.append(loss.item())
  val_losses.append(val_loss.item())

  # 保存最佳模型
  if val_loss.item() < best_val_loss:
    best_val_loss = val_loss.item()
    best_epoch = epoch + 1
    torch.save(model.state_dict(), temp_path)

  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

# 加载最佳模型
# model.load_state_dict(torch.load(temp_path))

# 记录结束时间
end_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())

print(f"验证损失的最小值: {best_val_loss:.6f} 出现在：{best_epoch}")
# 绘制训练损失和验证损失的变化曲线
plt.figure(figsize=(6,4))
plt.plot(train_losses,label='train loss')
plt.plot(val_losses,label='vall loss')
plt.title(f'Loss Curve best val_loss:{best_val_loss:.6f}')
plt.legend()
plt.savefig(base_path +'/loss ' + str(best_val_loss) + '.png')
plt.show()

# 在测试集上进行预测
model.eval()
with torch.no_grad():
  y_pred = model(x_test).cpu().numpy()
  # y_pred = model(x_val).cpu().numpy()

# 反标准化预测值和真实值
y_pred_inv = trips_transformer.inverse_transform(y_pred.reshape(1, -1))
y_test_inv = trips_transformer.inverse_transform(y_test.cpu().numpy().reshape(1, -1))

# 计算均方根误差
rmse_lstm = round(np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())), 6)

# 计算平均绝对百分比误差（MAPE）
mape_lstm = round(np.mean(np.abs((y_test_inv.flatten() - y_pred_inv.flatten()) / y_test_inv.flatten())) * 100, 2)

# 计算加权平均百分比误差（WAPE）
wape_lstm = round(np.sum(np.abs(y_test_inv.flatten() - y_pred_inv.flatten())) / np.sum(np.abs(y_test_inv.flatten())) * 100, 2)

# 计算决定系数（R²）
r2_lstm = round(r2_score(y_test_inv.flatten(), y_pred_inv.flatten()), 6)

print(f"RMSE: {rmse_lstm}")
print(f"MAPE: {mape_lstm}%")
print(f"WAPE: {wape_lstm}%")
print(f"R²: {r2_lstm}")

# 绘制预测结果
plt.figure(figsize=(12, 4))
plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), marker='.', label="pred")
plt.title(f'LSTM Prediction RMSE: {rmse_lstm}, MAPE: {mape_lstm}%, WAPE: {wape_lstm}%, R²: {r2_lstm}')
plt.legend()
plt.savefig(base_path + f'{rmse_lstm}_LSTM.png')
plt.show()

torch.save(model.state_dict(), base_path + 'bike_pred_model.pth')

# 记录数据指标
new_df = pd.DataFrame([[start_time,end_time,data_name,'pytorch',0,train_percentage,time_steps,hidden_size1,dropout1,hidden_size2,dropout2,2000,batch_size,rmse_lstm,best_val_loss]],columns=['start_time','end_time','data_name','kuangjia','index','train_percentage','time_steps','l1','d1','l2','d2','epochs','batch_size','rmse_lstm','min_val_loss'])
save_data = train_df._append(new_df)
save_data.to_excel('train.xlsx',index=False)
print('数据记录完成')