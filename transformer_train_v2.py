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
sns.set_style("darkgrid")
torch.manual_seed(42)
np.random.seed(42)

# 硬件检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BikeDataset(Dataset):
    def __init__(self, data, feature_cols, target_col, time_steps=7, future_steps=1):
        self.X, self.y = self.create_sequences(data, feature_cols, target_col, time_steps, future_steps)
        
    def create_sequences(self, data, feature_cols, target_col, time_steps, future_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps + 1):
            X.append(data[feature_cols].iloc[i:i+time_steps].values)
            y.append(data[target_col].iloc[i+time_steps:i+time_steps+future_steps].values)
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4, ff_dim=256, num_layers=3, 
                 dropout=0.1, time_steps=7):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, time_steps)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, 1)
        )
        
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, mask)
        x = torch.mean(x, dim=1)  # Global average pooling
        return self.decoder(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def process_data(data_path):
    # 读取数据
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    
    # 首先处理原始数据中的NaN值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 特征工程
    # df['year'] = df.index.year
    # df['month'] = df.index.month
    # df['day'] = df.index.day
    # df['dayofweek'] = df.index.dayofweek
    
    # 添加周期性特征
    # df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    # df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    # df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    # df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    # df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    # df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # 滑动统计特征 - 添加最小值填充
    # df['trips_7d_mean'] = df['trips'].rolling(7, min_periods=1).mean()
    # df['trips_7d_std'] = df['trips'].rolling(7, min_periods=1).std()
    
    # 再次检查并填充任何剩余的NaN值
    # df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 验证没有NaN值
    # if df.isna().any().any():
    #     raise ValueError("Data still contains NaN values after processing")
    
    return df

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, 
                      scheduler, num_epochs, device, patience=20):
    best_val_loss = float('inf')
    # patience_counter = 0
    train_losses = []
    val_losses = []
    train_rmse = []  # 新增
    val_rmse = []    # 新增
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []  # 新增
        train_trues = []  # 新增
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
            # 收集预测值和真实值用于计算RMSE
            train_preds.extend(outputs.cpu().detach().numpy())
            train_trues.extend(batch_y.cpu().numpy())
            
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []  # 新增
        val_trues = []  # 新增
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                
                # 收集预测值和真实值用于计算RMSE
                val_preds.extend(outputs.cpu().numpy())
                val_trues.extend(batch_y.cpu().numpy())
        
        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 计算RMSE
        current_train_rmse = np.sqrt(mean_squared_error(train_trues, train_preds))
        current_val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
        
        # 记录损失和RMSE
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_rmse.append(current_train_rmse)
        val_rmse.append(current_val_rmse)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './model/temp/temp.pth')
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train RMSE: {current_train_rmse:.4f}, Val RMSE: {current_val_rmse:.4f}')
    
    return train_losses, val_losses, train_rmse, val_rmse

def objective(trial):
    # 定义超参数搜索空间
    num_heads = trial.suggest_int("num_heads", 2, 8, step=2)
    d_model = trial.suggest_int("d_model", num_heads * 8, num_heads * 32, step=num_heads * 8)  # 确保 d_model 是 num_heads 的倍数
    ff_dim = trial.suggest_int("ff_dim", 128, 512, step=128)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    
    # 数据准备
    df = process_data('./data/daily_citi_bike_trip_counts_and_weather.csv')
    feature_cols = [
        'precipitation',
        'snow_depth',
        'snowfall',
        'max_t',
        'min_t',
        'average_wind_speed',
        'dow',
        'year',
        'month',
        # 'stations_in_service',
        'weekday',
        'weekday_non_holiday',
        'dt',
        'season'
    ]
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    target_scaler = RobustScaler()
    df['trips'] = target_scaler.fit_transform(df[['trips']])
    
    TIME_STEPS = 1
    BATCH_SIZE = 32
    NUM_EPOCHS = 50  # 调优时减少训练轮数以加快速度
    dataset = BikeDataset(df, feature_cols, 'trips', TIME_STEPS)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.8 * len(dataset))
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, val_size))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 模型初始化
    model = ImprovedTransformer(
        input_dim=len(feature_cols),
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        time_steps=TIME_STEPS
    ).to(device)
    
    # 优化配置
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10, verbose=False)
    
    # 训练模型
    train_losses, val_losses, train_rmse, val_rmse = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        NUM_EPOCHS, device)
    
    # 返回验证集的最终RMSE作为目标值
    return val_rmse[-1]

def main():
    # 使用Optuna进行超参数调优
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # 运行20次实验
    
    # 输出最佳超参数
    print("Best hyperparameters:", study.best_params)
    
    # 使用最佳超参数重新训练模型
    best_params = study.best_params
    d_model = best_params["d_model"]
    num_heads = best_params["num_heads"]
    ff_dim = best_params["ff_dim"]
    num_layers = best_params["num_layers"]
    dropout = best_params["dropout"]
    learning_rate = best_params["learning_rate"]
    
    # 数据准备
    df = process_data('./data/daily_citi_bike_trip_counts_and_weather.csv')
    
    # 特征选择
    feature_cols = [
        'precipitation',
        'snow_depth',  
        'snowfall',
        'max_t',
        'min_t',
        'average_wind_speed',
        'dow',
        'year',
        'month',
        # 'stations_in_service',
        'weekday',
        'weekday_non_holiday',
        'dt',
        'season'
    ]

    # 数据标准化
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    target_scaler = RobustScaler()
    df['trips'] = target_scaler.fit_transform(df[['trips']])
    
    # 模型参数
    TIME_STEPS = 1
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    
    # 创建数据集
    dataset = BikeDataset(df, feature_cols, 'trips', TIME_STEPS)
    train_size = int(0.7 * len(dataset))
    test_size = int(0.85 * len(dataset))
    
    # 方式2：按时间顺序分割
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, test_size))
    
    # 注意：只在训练集使用shuffle=True
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False)  # 训练集是否打乱由SHUFFLE_SPLIT控制
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False)  # 验证集永远不打乱
    
    # 模型初始化
    model = ImprovedTransformer(
        input_dim=len(feature_cols),
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        time_steps=TIME_STEPS
    ).to(device)
    
    # 训练配置
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=10, 
                                                   verbose=True)
    
    # 训练模型
    train_losses, val_losses, train_rmse, val_rmse = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        NUM_EPOCHS, device)
    
    # 保存结果
    save_dir = f'./model/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 模型评估
    model.eval()
    val_preds = []
    val_trues = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            val_preds.extend(outputs.cpu().numpy())
            val_trues.extend(batch_y.numpy())
    
    # 反标准化预测结果
    val_preds = target_scaler.inverse_transform(np.array(val_preds))
    val_trues = target_scaler.inverse_transform(np.array(val_trues).reshape(-1, 1))
    
    # 计算RMSE
    final_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
        
    # 损失曲线
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/loss.png')
    plt.show()
    
    # 预测结果对比
    plt.figure(figsize=(12, 4))
    plt.plot(val_trues, label='True Values',marker='.', alpha=0.7)
    plt.plot(val_preds, label='Predictions',marker='.', alpha=0.7)
    plt.title(f'Predictions vs True Values (RMSE={final_rmse:.2f})')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Bike Trips')
    plt.legend()
    
    # plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png')
    plt.show()
    
    # # 保存最终的指标
    # with open(f'{save_dir}/metrics.txt', 'w') as f:
    #     f.write(f'Final RMSE: {final_rmse:.4f}\n')
    #     f.write(f'Final Training Loss: {train_losses[-1]:.4f}\n')
    #     f.write(f'Final Validation Loss: {val_losses[-1]:.4f}\n')
    
    # 保存完整模型和参数
    torch.save(model.state_dict(), f'{save_dir}/final_model_state.pth')
    torch.save(model, f'{save_dir}/full_model.pth')
    
    # 保存标准化器
    torch.save({
        'feature_scaler': scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'time_steps': TIME_STEPS
    }, f'{save_dir}/model_config.pth')

if __name__ == "__main__":
    main()
