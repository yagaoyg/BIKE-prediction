import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import os
from base_models import PureLSTM, PureCNN, PureTransformer

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_preprocess_data(data_path, train_split=0.5, val_split=0.6, test_split=0.7):
    # 加载数据
    data = pd.read_csv(data_path,  index_col=['datetime'])
    
    # 划分数据集
    train_size = int(len(data) * train_split)
    val_size = int(len(data) * val_split)
    test_size = int(len(data) * test_split)
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:val_size]
    test_data = data.iloc[val_size:test_size]
    # test_data = data.iloc[train_size:val_size]
    
    # 特征标准化
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    # 选择特征
    feature_cols = [
        'season', 'year', 'month', 'day', 'dow', 'hour',
        'holiday', 'workingday', 'weather', 'temp',
        'atemp', 'humidity', 'windspeed','regrate'
    ]
    
    # 标准化特征
    train_data.loc[:, feature_cols] = feature_scaler.fit_transform(train_data[feature_cols])
    val_data.loc[:, feature_cols] = feature_scaler.transform(val_data[feature_cols])
    test_data.loc[:, feature_cols] = feature_scaler.transform(test_data[feature_cols])
    
    # 标准化目标值
    train_data.loc[:, 'count'] = target_scaler.fit_transform(train_data[['count']])
    val_data.loc[:, 'count'] = target_scaler.transform(val_data[['count']])
    test_data.loc[:, 'count'] = target_scaler.transform(test_data[['count']])
    
    return (train_data, val_data, test_data), (feature_scaler, target_scaler), feature_cols

def create_sequences(data, feature_cols, target_col, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[feature_cols].iloc[i:(i + seq_length)].values
        y = data[target_col].iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return torch.FloatTensor(xs), torch.FloatTensor(ys)

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, model_save_path, patience=50):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs.squeeze(), batch_y).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停: epoch {epoch}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses, best_val_loss

def evaluate_model(model, data_loader, criterion, target_scaler, device):
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            total_loss += loss.item()
            
            # 收集预测值和实际值
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    # 反标准化
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    wape = np.sum(np.abs(actuals - predictions)) / np.sum(np.abs(actuals)) * 100
    r2 = r2_score(actuals, predictions)
    
    return {
        'loss': total_loss / len(data_loader),
        'rmse': rmse,
        'mape': mape,
        'wape': wape,
        'r2': r2
    }, predictions, actuals

def main():
    # 数据准备
    data_path = './data/china_bike_data_2022.csv'
    (train_data, val_data, test_data), (feature_scaler, target_scaler), feature_cols = load_and_preprocess_data(data_path)
    
    # 创建序列数据
    seq_length = 12
    batch_size = 64
    
    # 准备数据加载器
    x_train, y_train = create_sequences(train_data, feature_cols, 'count', seq_length)
    x_val, y_val = create_sequences(val_data, feature_cols, 'count', seq_length)
    x_test, y_test = create_sequences(test_data, feature_cols, 'count', seq_length)
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    # 模型参数
    input_size = len(feature_cols)
    models = {
        'LSTM': PureLSTM(input_size=input_size),
        'CNN': PureCNN(input_size=input_size, seq_len=seq_length),
        'Transformer': PureTransformer(input_size=input_size)
    }
    
    # 训练和评估每个模型
    results = {}
    base_path = f"./base_model/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(base_path, exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\n训练 {model_name} 模型...")
        model = model.to(device)
        
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model_save_path = os.path.join(base_path, f"{model_name}_best.pth")
        
        # 训练模型
        train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=400, device=device, model_save_path=model_save_path
        )
        
        # 加载最佳模型
        model.load_state_dict(torch.load(model_save_path))
        
        # 评估模型
        test_metrics, predictions, actuals = evaluate_model(
            model, test_loader, criterion, target_scaler, device
        )
        
        results[model_name] = {
            'test_metrics': test_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions,
            'actuals': actuals
        }
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{model_name} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(base_path, f"{model_name}_loss.png"))
        plt.close()
        
        # 绘制预测结果
        plt.figure(figsize=(15, 5))
        plt.plot(actuals, label='Actual', alpha=0.5, marker='.')
        plt.plot(predictions, label='Predicted', alpha=0.5, marker='.')
        plt.title(f'{model_name} Predictions vs Actuals\nRMSE: {test_metrics["rmse"]:.2f}, MAPE: {test_metrics["mape"]:.2f}%, WAPE:{test_metrics["wape"]:.2f},R²: {test_metrics["r2"]:.4f}')
        plt.xlabel('Time')
        plt.ylabel('count')
        plt.legend()
        plt.savefig(os.path.join(base_path, f"{model_name}_predictions.png"))
        plt.close()
        
        print(f"\n{model_name} 测试结果:")
        print(f"RMSE: {test_metrics['rmse']:.2f}")
        print(f"MAPE: {test_metrics['mape']:.2f}%")
        print(f"WAPE: {test_metrics['wape']:.2f}%")
        print(f"R²: {test_metrics['r2']:.4f}")
    
    # 保存所有模型的评估结果
    results_df = pd.DataFrame({
        model_name: {
            'RMSE': metrics['test_metrics']['rmse'],
            'MAPE': metrics['test_metrics']['mape'],
            'WAPE': metrics['test_metrics']['wape'],
            'R²': metrics['test_metrics']['r2']
        }
        for model_name, metrics in results.items()
    }).T
    
    results_df.to_csv(os.path.join(base_path, 'model_comparison.csv'))
    print("\n模型比较结果已保存到:", os.path.join(base_path, 'model_comparison.csv'))

if __name__ == "__main__":
    main()