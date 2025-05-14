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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style("darkgrid")

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
    
data_name = 'china_bike_data_2022'

# 引入数据集
data = pd.read_csv('./data/' + data_name + '.csv',
                   parse_dates=['datetime'],
                   index_col=['datetime'])

# print(data.head())

train_percentage = 0.5
val_percentage = 0.6
test_percentage = 0.7

train_size = int(len(data) * train_percentage)
val_size = int(len(data) * val_percentage)
test_size = int(len(data) * test_percentage)

train_data = data.iloc[0:train_size]
val_data = data.iloc[train_size:val_size]
test_data = data.iloc[val_size:test_size]
# print(len(train_data), len(test_data))

# 选取特征
cols = ['season',
        'year',
        'month',
        'day',
        'dow',
        'hour',
        'holiday',
        'workingday',
        'weather',
        'temp',
        'atemp',
        'humidity',
        'windspeed',
        'regrate'
        ]

# 特征量标准化
transformer = RobustScaler()
transformer = transformer.fit(train_data[cols].to_numpy())

train_data.loc[:, cols] = transformer.transform(train_data[cols].to_numpy())
val_data.loc[:, cols] = transformer.transform(val_data[cols].to_numpy())
test_data.loc[:, cols] = transformer.transform(test_data[cols].to_numpy())

# 目标量标准化
count_transformer = RobustScaler()
count_transformer = count_transformer.fit(train_data[['count']])

train_data.loc[:, 'count'] = count_transformer.transform(train_data[['count']])
val_data.loc[:, 'count'] = count_transformer.transform(val_data[['count']])
test_data.loc[:, 'count'] = count_transformer.transform(test_data[['count']])

# 将输入的时序数据 x 和标签 y 转换成适合 LSTM 模型训练的数据格式
def create_dataset(x, y, time_steps=1):
  xs, ys = [], []
  for i in range(len(x) - time_steps):
      v = x.iloc[i:(i + time_steps)].values
      xs.append(v)
      ys.append(y.iloc[i + time_steps])
  return np.array(xs), np.array(ys)

time_steps = 24  # 24小时的时间步长
batch_size = 64

x_train, y_train = create_dataset(train_data, train_data['count'], time_steps)
x_val, y_val = create_dataset(val_data, val_data['count'], time_steps)
x_test, y_test = create_dataset(test_data, test_data['count'], time_steps)

# 转换为 PyTorch Tensor
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

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


# 设置超参数
input_size = x_train.shape[2]
hidden_size1 = 128
hidden_size2 = 80
dropout1 = 0.4
dropout2 = 0.3
epochs = 100
learning_rate = 0.0001

# 创建自定义的LSTM模型
model = MYLSTMModel(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

# 定义损失函数和优化器
criterion = nn.HuberLoss()

# 加载最佳模型
model.load_state_dict(torch.load('./best_model/cnn-bilstm/bike_pred_model.pth'))

# 在测试集上进行预测
print("\n开始测试集评估...")
model.eval()
with torch.no_grad():
    test_outputs = model(x_test)
    test_loss = criterion(test_outputs.squeeze(), y_test)
    print(f'\nTest Loss: {test_loss.item():.6f}')

# 对测试集预测结果进行反标准化处理
test_preds = test_outputs.cpu().numpy()
test_preds = count_transformer.inverse_transform(test_preds.reshape(1, -1))
test_trues = y_test.cpu().numpy()
test_trues = count_transformer.inverse_transform(test_trues.reshape(1, -1))

# 计算测试集评估指标
rmse_test = round(np.sqrt(mean_squared_error(test_trues.flatten(), test_preds.flatten())), 6)
mape_test = round(np.mean(np.abs((test_trues.flatten() - test_preds.flatten()) / test_trues.flatten())) * 100, 2)
wape_test = round(np.sum(np.abs(test_trues.flatten() - test_preds.flatten())) / np.sum(np.abs(test_trues.flatten())) * 100, 2)
r2_test = round(r2_score(test_trues.flatten(), test_preds.flatten()), 6)

print(f"\n测试集性能评估:")
print(f"RMSE: {rmse_test}")
print(f"MAPE: {mape_test}%")
print(f"WAPE: {wape_test}%")
print(f"R²: {r2_test}")

# 绘制测试集预测结果对比图
plt.figure(figsize=(12, 4))
plt.plot(test_trues.flatten(), marker='.', label="true")
plt.plot(test_preds.flatten(), marker='.', label="pred")
plt.title(f'Test Set Prediction\nRMSE: {rmse_test}, MAPE: {mape_test}%, WAPE: {wape_test}%, R²: {r2_test}')
plt.legend()
plt.savefig(base_path + f'/test_prediction_{rmse_test}.png')
plt.show()

# 保存模型和数据记录
torch.save(model.state_dict(), base_path + 'bike_pred_model.pth')

torch.save(model, f'{base_path}/full_model.pth')