import torch
import torch.nn as nn

# 纯LSTM模型
class PureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(PureLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2是因为是双向LSTM
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # 取最后一个时间步
        out = self.fc(out)
        return out

# 纯CNN模型
class PureCNN(nn.Module):
    def __init__(self, input_size, seq_len):
        super(PureCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度顺序为 (batch, channels, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(-1)  # 移除最后一个维度
        out = self.fc(x)
        return out

# 纯Transformer模型
class PureTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(PureTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)  # 投影到transformer维度
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 对序列取平均
        x = self.dropout(x)
        out = self.output_layer(x)
        return out
      
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
  
class HierarchicalFeatureProcessor(nn.Module):
    """
    层次化特征处理模块，用于对输入特征进行分层处理。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 第一层全连接
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout)  # Dropout 防止过拟合
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),  # 第二层全连接
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 前向传播
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ImprovedTransformer(nn.Module):
    """
    改进的 Transformer 模型，支持层次化特征处理和注意力权重输出。
    """
    def __init__(self, input_dim, d_model=64, num_heads=4, ff_dim=256, num_layers=3, 
                 dropout=0.1, time_steps=7):
        super().__init__()
        self.feature_processor = HierarchicalFeatureProcessor(input_dim, d_model, d_model, dropout)  # 层次化特征处理
        self.input_proj = nn.Linear(d_model, d_model)  # 输入投影层
        self.pos_encoder = PositionalEncoding(d_model, time_steps)  # 位置编码
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 多层编码器
        
        self.attention_weights = nn.Linear(d_model, 1)  # 注意力权重计算层
        self.decoder = nn.Sequential(
            nn.Linear(d_model, ff_dim),  # 解码器第一层
            nn.GELU(),  # 激活函数
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),  # 解码器第二层
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, 1)  # 输出层
        )
        
    def forward(self, x, mask=None):
        # 前向传播
        x = self.feature_processor(x)  # 层次化特征处理
        x = self.input_proj(x)  # 输入投影
        x = self.pos_encoder(x)  # 添加位置编码
        x = self.transformer(x, mask)  # Transformer 编码器
        
        # 计算注意力权重并加权求和
        attention_scores = torch.softmax(self.attention_weights(x), dim=1) 
        x = torch.sum(x * attention_scores, dim=1)
        
        return self.decoder(x), attention_scores  # 返回解码结果和注意力权重

class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于为输入添加位置信息。
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # 频率因子
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        pe = pe.unsqueeze(0)  # 添加批次维度
        self.register_buffer('pe', pe)  # 注册为缓冲区，不参与梯度计算

    def forward(self, x):
        # 将位置编码添加到输入
        return x + self.pe[:, :x.size(1)]