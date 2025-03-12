import torch
import torch.nn as nn

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# 自定义LSTM模型
# 双向LSTM + CNN
class MYLSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, dropout1, dropout2):
    super(MYLSTMModel, self).__init__()
    self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
    self.lstm1 = nn.LSTM(128, hidden_size1, batch_first=True, bidirectional=True)
    self.dropout1 = nn.Dropout(dropout1)
    self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
    self.dropout2 = nn.Dropout(dropout2)
    self.fc = nn.Linear(hidden_size2 * 2, 1)

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


# 设置超参数
input_size = 14
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
# model = BiLSTMModel(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)

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
model = MYLSTMModel(input_size, hidden_size1, hidden_size2, dropout1, dropout2).to(device)
