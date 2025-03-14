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
from sklearn.metrics import mean_squared_error

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

train_percentage = 0.9
train_size = int(len(data) * train_percentage)
test_size = len(data) - train_size
train_data,test_data = data.iloc[0:train_size],data.iloc[train_size:len(data)]
# print(len(train_data), len(test_data))

# 选取特征
cols = ['season',
        'holiday',
        'workingday',
        'weather',
        'temp',
        'atemp',
        'humidity',
        'windspeed']

# 特征量标准化
transformer = RobustScaler()
transformer = transformer.fit(train_data[cols].to_numpy())

train_data.loc[:, cols] = transformer.transform(train_data[cols].to_numpy())
test_data.loc[:, cols] = transformer.transform(test_data[cols].to_numpy())

# 目标量标准化
count_transformer = RobustScaler()
count_transformer = count_transformer.fit(train_data[['count']])

train_data.loc[:, 'count'] = count_transformer.transform(train_data[['count']])
test_data.loc[:, 'count'] = count_transformer.transform(test_data[['count']])

# 将输入的时序数据 x 和标签 y 转换成适合 LSTM 模型训练的数据格式
def create_dataset(x, y, time_steps=1):
  xs, ys = [], []
  for i in range(len(x) - time_steps):
      v = x.iloc[i:(i + time_steps)].values
      xs.append(v)
      ys.append(y.iloc[i + time_steps])
  return np.array(xs), np.array(ys)

time_steps = 1

x_train, y_train = create_dataset(train_data, train_data['count'], time_steps)
x_test, y_test = create_dataset(test_data, test_data['count'], time_steps)

# 转换为 PyTorch Tensor
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# 基于当前时间创建路径 作为基础路径使用
base_path = "./model/{0:%Y-%m-%d %H-%M-%S}/".format(datetime.now())

# 临时路径 用于保存训练过程中回调函数保存的模型
temp_path = './model/temp/temp_bike_pred_model.pth'

# 引入数据指标记录表
train_df = pd.read_excel('train.xlsx')

# 记录开始时间
start_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())

# 设置保存模型的路径
model_save_path = base_path + '/bike_pred_model.pth'

# 创建保存模型的文件夹（如果没有的话）
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
    super(LSTMModel, self).__init__()
    self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
    self.dropout1 = nn.Dropout(dropout1)
    self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
    self.dropout2 = nn.Dropout(dropout2)
    self.fc = nn.Linear(hidden_size2, 1)

  def forward(self, x):
    out, _ = self.lstm1(x)
    out = self.dropout1(out)
    out, _ = self.lstm2(out)
    out = self.dropout2(out)
    out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
    return out

# 双向LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
        super(BiLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_size2 * 2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# CNN和LSTM结合模型
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.lstm1 = nn.LSTM(128, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_size, time_steps)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch_size, time_steps, 128)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 在 LSTM 层之间添加 Batch Normalization 的模型
class LSTMModelWithBN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
        super(LSTMModelWithBN, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.bn1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.bn2(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 使用三层LSTM的模型
class TriLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, dropout1, dropout2, dropout3):
        super(TriLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.dropout3 = nn.Dropout(dropout3)
        self.fc = nn.Linear(hidden_size3, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 有注意力机制的LSTM模型
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
        super(AttentionLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_size2, 1)
        self.attention = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        # Attention weights
        attention_weights = torch.softmax(self.attention(out), dim=1)
        context_vector = torch.sum(attention_weights * out, dim=1)
        out = self.fc(context_vector)
        return out

# 使用 LeakyReLu 激活函数的 LSTM 模型
class LSTMModelWithLeakyReLU(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
        super(LSTMModelWithLeakyReLU, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_size2, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.leaky_relu(out)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.leaky_relu(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 使用残差链接的LSTM模型
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
        super(ResidualLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_size2, 1)
        self.residual_fc = nn.Linear(input_size, hidden_size2)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        residual = self.residual_fc(x[:, -1, :]).unsqueeze(1)
        out = out + residual
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 自己设置的LSTM模型
class MYLSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
    super(MYLSTMModel, self).__init__()
    self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True,bidirectional=True)
    self.dropout1 = nn.Dropout(dropout1)
    self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True,bidirectional=True)
    self.dropout2 = nn.Dropout(dropout2)
    self.fc = nn.Linear(hidden_size2, 1)
    self.attention = nn.Linear(hidden_size2, 1)

  def forward(self, x):
    out, _ = self.lstm1(x)
    out = self.dropout1(out)
    out, _ = self.lstm2(out)
    out = self.dropout2(out)
    out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
    return out


# 设置超参数
input_size = x_train.shape[2]
hidden_size1 = 144
hidden_size2 = 96
dropout1 = 0.4
dropout2 = 0.3
epochs = 2000
batch_size = 128
learning_rate = 0.001

# 创建模型并移动到设备
# model = LSTMModel(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

# 创建双向LSTM模型 
model = BiLSTMModel(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

# 创建CNN和LSTM结合模型
# model = CNNLSTMModel(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

# 创建LSTM模型与BN结合模型
# model = LSTMModelWithBN(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

# 创建三层LSTM模型
# model = TriLSTMModel(input_size, hidden_size1, hidden_size2, 40, dropout1, dropout2, 0.3).to(device)

# 创建有注意力机制的LSTM模型
# model = AttentionLSTM(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

# 创建使用LeakyReLU激活函数的LSTM模型
# model = LSTMModelWithLeakyReLU(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

# 创建使用残差链接的LSTM模型
# model = ResidualLSTM(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

# 创建自定义的LSTM模型
# model = MYLSTMModel(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

# 训练模型
train_losses, val_losses = [], []
best_val_loss = float('inf')

# patience = 50
# trigger_times = 0
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

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
    val_outputs = model(x_test)
    val_loss = criterion(val_outputs.squeeze(), y_test)

  train_losses.append(loss.item())
  val_losses.append(val_loss.item())

  # 保存最佳模型
  if val_loss.item() < best_val_loss:
    best_val_loss = val_loss.item()
    torch.save(model.state_dict(), temp_path)
    trigger_times = 0
  # else:
  #   trigger_times += 1
  #   if trigger_times >= patience:
  #     print(f"Early stopping at epoch {epoch+1}")
  #     break
    
  # scheduler.step(val_loss)

  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

# 加载最佳模型
model.load_state_dict(torch.load(temp_path))

# 记录结束时间
end_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())

print(f"验证损失的最小值: {best_val_loss:.6f}")
# 绘制训练损失和验证损失的变化曲线
plt.figure(figsize=(6,4))
plt.plot(train_losses,label='train loss')
plt.plot(val_losses,label='vall loss')
plt.legend()
plt.savefig(base_path +'/loss ' + str(best_val_loss) + '.png')
plt.show()

# 在测试集上进行预测
model.eval()
with torch.no_grad():
  y_pred = model(x_test).cpu().numpy()

# 反标准化预测值和真实值
y_pred_inv = count_transformer.inverse_transform(y_pred.reshape(1, -1))
y_test_inv = count_transformer.inverse_transform(y_test.cpu().numpy().reshape(1, -1))

# 计算均方根误差
rmse_lstm = round(np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())), 6)
print(f"RMSE: {rmse_lstm}")

# 绘制预测结果
plt.figure(figsize=(12, 4))
plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), marker='.', label="pred")
plt.title(f'LSTM Prediction RMSE: {rmse_lstm}')
plt.legend()
plt.savefig(base_path + str(rmse_lstm) + '_LSTM.png')
plt.show()

# 记录数据指标
new_df = pd.DataFrame([[start_time,end_time,data_name,'pytorch',0,train_percentage,time_steps,hidden_size1,dropout1,hidden_size2,dropout2,epochs,batch_size,rmse_lstm,best_val_loss]],columns=['start_time','end_time','data_name','kuangjia','index','train_percentage','time_steps','l1','d1','l2','d2','epochs','batch_size','rmse_lstm','min_val_loss'])
save_data = train_df._append(new_df)
save_data.to_excel('train.xlsx',index=False)
print('数据记录完成')