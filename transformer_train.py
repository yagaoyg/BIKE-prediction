# 自行车使用量预测PyTorch实现
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
from sklearn.metrics import mean_squared_error

# 配置设置
sns.set_style("darkgrid")
# torch.manual_seed(42)

# 硬件检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 自定义数据集类
class BikeDataset(Dataset):
    def __init__(self, data, feature_cols, target_col, time_steps=7):
        self.X, self.y = self.create_sequences(data, feature_cols, target_col, time_steps)
        
    def create_sequences(self, data, feature_cols, target_col, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[feature_cols].iloc[i:i+time_steps].values)
            y.append(data[target_col].iloc[i+time_steps])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Transformer模型类
class BikeTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=2, ff_dim=512, dropout=0.3, num_layers=3,time_steps=7):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 32)  # 维度提升
        self.pos_encoder = PositionalEncoding(d_model=32, max_len=time_steps)  # 新增
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)  # [batch, seq_len, 32]
        x = self.pos_encoder(x)  # 添加位置编码
        x = self.encoder(x)
        x = x.mean(dim=1)  # 时序平均池化
        return self.decoder(x)

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=7):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# 数据预处理函数
def load_and_preprocess(data_path='./data/daily_citi_bike_trip_counts_and_weather.csv'):
    raw_data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    
    # 特征工程
    feature_cols = [
        'precipitation', 'snow_depth', 'snowfall', 'max_t', 'min_t',
        'average_wind_speed', 'dow', 'year', 'month', 'holiday',
        # 'stations_in_service',
        'weekday', 'weekday_non_holiday', 'dt', 'season'
    ]
    target_col = 'trips'
    
    # 添加滑动窗口特征
    # raw_data['7d_avg_trips'] = raw_data[target_col].rolling(7).mean().fillna(0)
    # feature_cols.append('7d_avg_trips')
    
    # 数据集划分
    train_size = int(len(raw_data) * 0.9)
    train_data = raw_data.iloc[:train_size]
    test_data = raw_data.iloc[train_size:]
    
    # 标准化处理
    feature_scaler = RobustScaler().fit(train_data[feature_cols])
    target_scaler = RobustScaler().fit(train_data[[target_col]])
    
    train_data.loc[:, feature_cols] = feature_scaler.transform(train_data[feature_cols])
    test_data.loc[:, feature_cols] = feature_scaler.transform(test_data[feature_cols])
    train_data[target_col] = target_scaler.transform(train_data[[target_col]])
    test_data[target_col] = target_scaler.transform(test_data[[target_col]])
    
    return train_data, test_data, feature_cols, target_col, target_scaler

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, save_path):
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    # patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val.unsqueeze(1)).item() * X_val.size(0)
        
        # 记录损失
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            
        # 添加早停机制
        # if val_loss < best_loss:
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= 30:
        #         print("Early stopping triggered")
        #         break
        
        # 学习率调度
        # scheduler.step(val_loss)
        
        # 打印进度
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
    
    return history, best_loss

# 主程序流程

# 数据准备
train_data, test_data, feature_cols, target_col, target_scaler = load_and_preprocess()

# 创建数据集
TIME_STEPS = 1
train_dataset = BikeDataset(train_data, feature_cols, target_col, TIME_STEPS)
test_dataset = BikeDataset(test_data, feature_cols, target_col, TIME_STEPS)

# 创建数据加载器
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 模型配置
model = BikeTransformer(
    input_dim=len(feature_cols),
    num_heads=4,
    ff_dim=256,
    dropout=0.2,
    num_layers=2,
    time_steps=TIME_STEPS
).to(device)

# 优化配置
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, verbose=True
)

# 创建保存目录
base_path = f"./model/{datetime.now():%Y-%m-%d %H-%M-%S}"
os.makedirs(base_path, exist_ok=True)

# 模型训练
history, best_loss = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=2000,
    device=device,
    save_path=base_path
)

print(f'最小损失: {best_loss:.6f}')

# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(base_path, 'best_model.pth')))

# 模型评估
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test = X_test.to(device)
        outputs = model(X_test).cpu().numpy()
        y_pred.extend(outputs.squeeze())
        y_true.extend(y_test.numpy())

# 反标准化
y_pred_inv = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
y_test_inv = target_scaler.inverse_transform(np.array(y_true).reshape(-1, 1))

# 计算指标
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f'Test RMSE: {rmse:.2f}')

# 可视化损失曲线
plt.figure(figsize=(6,4))
plt.plot(history['train_loss'],label='train loss')
plt.plot(history['val_loss'],label='vall loss')
plt.legend()
plt.savefig(base_path +'/loss ' + str(best_loss) + '.png')
plt.show()

# 可视化结果
plt.figure(figsize=(12, 4))
plt.plot(y_test_inv, label='True', marker='.', alpha=0.7)
plt.plot(y_pred_inv, label='Pred', marker='.', alpha=0.7)
plt.title(f'Transformer Prediction (RMSE={rmse:.2f})')
plt.legend()
plt.savefig(os.path.join(base_path, 'prediction_comparison.png'))
plt.show()

# 保存完整模型
torch.save(model, os.path.join(base_path, 'full_model.pth'))