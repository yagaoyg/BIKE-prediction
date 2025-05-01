import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna  # 新增

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
        'precipitation', 
        'snow_depth', 
        'snowfall',
        'max_t', 
        'min_t', 
        'average_wind_speed',
        # 'dow', 
        # 'year', 
        'month', 
        'weekday',
        'weekday_non_holiday', 
        'dt', 
        'season'
    ],
    'target_col': 'trips',
    'time_steps': 7,  # 时间步长
    'batch_size': 32,
    'train_epochs': 400,
    'tuning_epochs': 50,
    'train_split': 0.7,     # 训练集比例 0-70%
    'val_split': 0.8,       # 验证集比例 70-80%
    'test_split': 0.9,      # 测试集比例 80-90%
    'model_params': {
        'd_model': 128,
        'num_heads': 4,
        'ff_dim': 256,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 1e-4
    }
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
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')  # 读取 CSV 数据
    df = df.fillna(method='ffill').fillna(method='bfill')  # 填充缺失值
    return df

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, 
                      scheduler, num_epochs, device, patience=20):
    """
    训练和评估模型。
    """
    best_val_loss = float('inf')  # 初始化最佳验证损失
    train_losses = []  # 记录训练损失
    val_losses = []  # 记录验证损失
    train_rmse = []  # 记录训练 RMSE
    val_rmse = []  # 记录验证 RMSE
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []  # 收集训练预测值
        train_trues = []  # 收集训练真实值
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_X)  # 修改为接收注意力权重
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()
            
            # 收集预测值和真实值用于计算 RMSE
            train_preds.extend(outputs.cpu().detach().numpy())
            train_trues.extend(batch_y.cpu().numpy())
            
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []  # 收集验证预测值
        val_trues = []  # 收集验证真实值
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, _ = model(batch_X)  # 修改为接收注意力权重
                val_loss += criterion(outputs, batch_y).item()
                
                # 收集预测值和真实值用于计算 RMSE
                val_preds.extend(outputs.cpu().numpy())
                val_trues.extend(batch_y.cpu().numpy())
        
        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 计算 RMSE
        current_train_rmse = np.sqrt(mean_squared_error(train_trues, train_preds))
        current_val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
        
        # 记录损失和 RMSE
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_rmse.append(current_train_rmse)
        val_rmse.append(current_val_rmse)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './model/temp/temp.pth')  # 保存最佳模型
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train RMSE: {current_train_rmse:.4f}, Val RMSE: {current_val_rmse:.4f}')
    
    return train_losses, val_losses, train_rmse, val_rmse

def objective(trial):
    """
    Optuna 超参数调优目标函数。
    """
    # 定义超参数搜索空间
    num_heads = trial.suggest_int("num_heads", 2, 8, step=2)
    d_model = trial.suggest_int("d_model", num_heads * 8, num_heads * 32, step=num_heads * 8)  # 确保 d_model 是 num_heads 的倍数
    ff_dim = trial.suggest_int("ff_dim", 128, 512, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    
    # 数据准备
    df = process_data('./data/daily_citi_bike_trip_counts_and_weather.csv')
    
    # 使用全局配置的特征和参数
    scaler = RobustScaler()
    df[CONFIG['feature_cols']] = scaler.fit_transform(df[CONFIG['feature_cols']])
    target_scaler = RobustScaler()
    df[CONFIG['target_col']] = target_scaler.fit_transform(df[[CONFIG['target_col']]])
    
    # 使用配置参数创建数据集
    dataset = BikeDataset(df, CONFIG['feature_cols'], CONFIG['target_col'], CONFIG['time_steps'])
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = int(CONFIG['val_split'] * len(dataset))
    test_size = int(CONFIG['test_split'] * len(dataset))
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(val_size, test_size))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 模型初始化
    model = ImprovedTransformer(
        input_dim=len(CONFIG['feature_cols']),
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        time_steps=CONFIG['time_steps']
    ).to(device)
    
    # 优化配置
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10, verbose=False)
    
    # 训练模型
    train_losses, val_losses, train_rmse, val_rmse = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        CONFIG['tuning_epochs'], device)
    
    # 返回验证集的最终 RMSE 作为目标值
    return val_rmse[-1]

def main():
    """
    主函数，执行超参数调优和模型训练。
    """
    # 使用 Optuna 进行超参数调优
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)  # 运行 20 次实验
    
    # 输出最佳超参数
    print("Best hyperparameters:", study.best_params)
    
    # 更新配置参数为最佳参数
    best_params = {
        'd_model': study.best_params['d_model'],
        'num_heads': study.best_params['num_heads'],
        'ff_dim': study.best_params['ff_dim'],
        'num_layers': study.best_params['num_layers'],
        'dropout': study.best_params['dropout']
    }
    CONFIG['model_params'].update(best_params)
    
    # 数据准备
    df = process_data('./data/daily_citi_bike_trip_counts_and_weather.csv')
    
    # 使用全局配置的特征和参数
    scaler = RobustScaler()
    df[CONFIG['feature_cols']] = scaler.fit_transform(df[CONFIG['feature_cols']])
    target_scaler = RobustScaler()
    df[CONFIG['target_col']] = target_scaler.fit_transform(df[[CONFIG['target_col']]])
    
    # 创建数据集
    dataset = BikeDataset(df, CONFIG['feature_cols'], CONFIG['target_col'], CONFIG['time_steps'])
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = int(CONFIG['val_split'] * len(dataset))
    test_size = int(CONFIG['test_split'] * len(dataset))
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(val_size, test_size))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 使用配置参数初始化模型
    model_params = CONFIG['model_params'].copy()
    learning_rate = model_params.pop('learning_rate')
    
    model = ImprovedTransformer(
        input_dim=len(CONFIG['feature_cols']),
        time_steps=CONFIG['time_steps'],
        **model_params
    ).to(device)
    
    # 训练配置
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=study.best_params['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=20, 
                                                   verbose=True)
    
    # 训练模型
    train_losses, val_losses, train_rmse, val_rmse = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        CONFIG['train_epochs'], device)
    
    # 保存结果
    save_dir = f'./model/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'
    os.makedirs(save_dir, exist_ok=True)
    
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
        
    # 损失曲线
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/loss.png')
    plt.show()
    
    # 预测结果对比
    plt.figure(figsize=(12, 4))
    plt.plot(val_trues, label='True Values', marker='.', alpha=0.7)
    plt.plot(val_preds, label='Predictions', marker='.', alpha=0.7)
    plt.title(f'Predictions vs True Values (RMSE={final_rmse:.2f}, MAPE={final_mape:.2f}%, WAPE={final_wape:.2f}%, R²={final_r2:.4f})')
    plt.legend()
    
    plt.savefig(f'{save_dir}/training_results.png')
    plt.show()
    
    # 保存完整模型和参数
    torch.save(model.state_dict(), f'{save_dir}/final_model_state.pth')
    torch.save(model, f'{save_dir}/full_model.pth')
    
    # 保存标准化器
    torch.save({
        'feature_scaler': scaler,
        'target_scaler': target_scaler,
        'feature_cols': CONFIG['feature_cols'],
        'time_steps': CONFIG['time_steps']
    }, f'{save_dir}/model_config.pth')

    # 在测试集上进行预测和评估
    print("\n开始测试集评估...")
    model.eval()
    test_loss = 0
    test_preds = []
    test_trues = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs, attention_scores = model(batch_X)  # 获取注意力权重
            outputs = outputs.squeeze()  # 确保输出维度正确
            test_preds.extend(outputs.cpu().numpy())
            test_trues.extend(batch_y.numpy())
            
    # 转换为 numpy 数组并重塑
    test_preds = np.array(test_preds).reshape(-1, 1)
    test_trues = np.array(test_trues).reshape(-1, 1)
    
    test_loss /= len(test_loader)
    test_preds = target_scaler.inverse_transform(np.array(test_preds))
    test_trues = target_scaler.inverse_transform(np.array(test_trues))
    
    # 计算测试集评估指标
    rmse_test = np.sqrt(mean_squared_error(test_trues, test_preds))
    mape_test = np.mean(np.abs((test_trues - test_preds) / test_trues)) * 100
    wape_test = np.sum(np.abs(test_trues - test_preds)) / np.sum(np.abs(test_trues)) * 100
    r2_test = r2_score(test_trues, test_preds)
    
    print(f"\n测试集性能评估:")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"RMSE: {rmse_test:.4f}")
    print(f"MAPE: {mape_test:.2f}%")
    print(f"WAPE: {wape_test:.2f}%")
    print(f"R²: {r2_test:.4f}")
    
    # 绘制测试集预测结果
    plt.figure(figsize=(12, 4))
    plt.plot(test_trues, label='True Values', marker='.', alpha=0.7)
    plt.plot(test_preds, label='Predictions', marker='.', alpha=0.7)
    plt.title(f'Test Set Predictions\nRMSE={rmse_test:.2f}, MAPE={mape_test:.2f}%, WAPE={wape_test:.2f}%, R²={r2_test:.4f}')
    plt.legend()
    plt.savefig(f'{save_dir}/test_results.png')
    plt.show()

if __name__ == "__main__":
    main()
