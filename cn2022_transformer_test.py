import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置设置
sns.set_style("darkgrid")  # 设置 Seaborn 图表样式
torch.manual_seed(42)  # 设置 PyTorch 随机种子
np.random.seed(42)  # 设置 NumPy 随机种子

# 硬件检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有 GPU 可用
print(f"Using device: {device}")  # 打印使用的设备信息

# 全局配置参数
CONFIG = {
    'feature_cols': [
        'season', 'year', 'month', 'day', 'dow', 'hour',
        'holiday', 'workingday', 'weather', 'temp',
        'atemp', 'humidity', 'windspeed'
    ],
    'target_col': 'count',
    'time_steps': 12,
    'batch_size': 64,
    'train_split': 0.6,
    'val_split': 0.7,
}

class BikeDataset(Dataset):
    """
    自定义数据集类，用于处理时间序列数据。
    """
    def __init__(self, data, feature_cols, target_col, time_steps=7, future_steps=1):
        # 创建时间序列数据
        self.X, self.y = self.create_sequences(data, feature_cols, target_col, time_steps, future_steps)
        
    def create_sequences(self, data, feature_cols, target_col, time_steps, future_steps):
        """
        根据时间步长和未来步长生成输入和目标序列。
        """
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps + 1):
            X.append(data[feature_cols].iloc[i:i+time_steps].values)  # 提取特征序列
            y.append(data[target_col].iloc[i+time_steps:i+time_steps+future_steps].values)  # 提取目标序列
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    
    def __len__(self):
        # 返回数据集的样本数量
        return len(self.X)
    
    def __getitem__(self, idx):
        # 根据索引返回样本
        return self.X[idx], self.y[idx]

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

def process_data(data_path):
    """
    数据预处理函数，读取数据并处理缺失值。
    """
    df = pd.read_csv(data_path, index_col='datetime')  # 读取 CSV 数据
    # df = df.fillna(method='ffill').fillna(method='bfill')  # 填充缺失值
    return df

def print_model_params(model):
    """打印模型的关键参数"""
    print("\n=== 模型参数信息 ===")
    
    # 获取特征处理器参数
    feature_processor = model.feature_processor
    print(f"特征处理器输入维度: {feature_processor.layer1[0].in_features}")
    print(f"特征处理器隐藏层维度: {feature_processor.layer1[0].out_features}")
    print(f"特征处理器输出维度: {feature_processor.layer2[0].out_features}")
    
    # 获取Transformer参数
    transformer_layer = model.transformer.layers[0]
    print(f"\nTransformer参数:")
    print(f"注意力头数: {transformer_layer.self_attn.num_heads}")
    print(f"模型维度(d_model): {transformer_layer.self_attn.embed_dim}")
    print(f"前馈网络维度: {transformer_layer.linear1.out_features}")
    print(f"Transformer层数: {len(model.transformer.layers)}")
    
    # 获取解码器参数
    decoder = model.decoder
    print(f"\n解码器参数:")
    print(f"输入维度: {decoder[0].in_features}")
    print(f"隐藏层维度: {decoder[0].out_features}")
    print(f"输出维度: {decoder[-1].out_features}")
    
    # 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

def main():
    """
    主函数，执行超参数调优和模型训练。
    """

    # 数据准备
    df = process_data('./data/china_bike_data_2022.csv')
    
    # 使用全局配置的特征和参数
    scaler = RobustScaler()
    df[CONFIG['feature_cols']] = scaler.fit_transform(df[CONFIG['feature_cols']])
    target_scaler = RobustScaler()
    df[CONFIG['target_col']] = target_scaler.fit_transform(df[[CONFIG['target_col']]])
    
    # 创建数据集
    dataset = BikeDataset(df, CONFIG['feature_cols'], CONFIG['target_col'], CONFIG['time_steps'])
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = int(CONFIG['val_split'] * len(dataset))
    
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, val_size))
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        
    # 保存结果
    save_dir = f'./model/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 最佳模型路径
    model_path = './best_model/3/full_model.pth'
    
    # 加载最佳模型
    model = torch.load(model_path).to(device)  # 加载模型到设备
    print("Model loaded successfully.")
    
    # 打印模型参数
    print_model_params(model)
    
    # 模型评估
    model.eval()
    val_preds = []
    val_trues = []
    attention_scores_list = []  # 新增
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs, attention_scores = model(batch_X)  # 获取注意力权重
            val_preds.extend(outputs.cpu().numpy())
            val_trues.extend(batch_y.numpy())
            attention_scores_list.append(attention_scores.cpu())  # 收集注意力权重
    
    # 反标准化预测结果
    val_preds = np.array(val_preds)  # 转换为 NumPy 数组
    val_trues = np.array(val_trues)  # 转换为 NumPy 数组

    val_preds = target_scaler.inverse_transform(val_preds)
    val_trues = target_scaler.inverse_transform(val_trues)
    
    # 计算 RMSE
    final_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
    
    # 计算平均绝对百分比误差（MAPE）
    final_mape = np.mean(np.abs((val_trues - val_preds) / val_trues)) * 100
    
    # 计算加权平均百分比误差（WAPE）
    final_wape = np.sum(np.abs(val_trues - val_preds)) / np.sum(np.abs(val_trues)) * 100
    
    # 计算决定系数（R²）
    final_r2 = r2_score(val_trues, val_preds)
    
    print(f"Final RMSE: {final_rmse:.4f}")
    print(f"Final MAPE: {final_mape:.2f}%")
    print(f"Final WAPE: {final_wape:.2f}%")
    print(f"Final R²: {final_r2:.4f}")
            
    # 预测结果对比
    plt.figure(figsize=(12, 4))
    plt.plot(val_trues, label='True Values', marker='.', alpha=0.7)
    plt.plot(val_preds, label='Predictions', marker='.', alpha=0.7)
    plt.title(f'Predictions vs True Values (RMSE={final_rmse:.2f}, MAPE={final_mape:.2f}%, WAPE={final_wape:.2f}%, R²={final_r2:.4f})')
    plt.legend()
    
    plt.savefig(f'{save_dir}/training_results.png')
    plt.show()

if __name__ == "__main__":
    main()
