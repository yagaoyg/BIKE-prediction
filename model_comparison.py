import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import os
from base_models import PureLSTM, PureCNN, PureTransformer,MYLSTMModel,ImprovedTransformer
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 全局特征列表
feature_cols = [
    'season', 'year', 'month', 'day', 'dow', 'hour',
    'holiday', 'workingday', 'weather', 'temp',
    'atemp', 'humidity', 'windspeed', 'regrate'
]

def load_and_preprocess_data(data_path, test_split=0.7):
    # 加载数据
    data = pd.read_csv(data_path, index_col=['datetime'])
    
    # 只使用测试数据
    test_size = int(len(data) * test_split)
    test_data = data.iloc[:test_size]
    
    # 特征标准化
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    # 标准化特征和目标值
    test_data.loc[:, feature_cols] = feature_scaler.fit_transform(test_data[feature_cols])
    test_data.loc[:, 'count'] = target_scaler.fit_transform(test_data[['count']])
    
    return test_data, feature_scaler, target_scaler

def create_sequences(data, feature_cols, target_col, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[feature_cols].iloc[i:(i + seq_length)].values
        y = data[target_col].iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return torch.FloatTensor(xs), torch.FloatTensor(ys)

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
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
    
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
    }, predictions.reshape(-1), actuals.reshape(-1)

def compare_models(models_path, test_loader, target_scaler, seq_length, device):
    model_files = [f for f in os.listdir(models_path) if f.endswith('.pth')]
    all_predictions = {}
    all_metrics = {}
    
    plt.figure(figsize=(15, 8))
    criterion = nn.HuberLoss()
    
    # 获取实际值
    actuals = None
    for batch_x, batch_y in test_loader:
        if actuals is None:
            actuals = []
        actuals.extend(batch_y.cpu().numpy())
    actuals = np.array(actuals)
    actuals = target_scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(-1)
    
    # 绘制实际值
    plt.plot(actuals, label='实际值', linewidth=2, alpha=0.7)
    
    # 对每个模型进行预测
    for model_file in model_files:
        model_name = model_file.replace('_best.pth', '')
        model_path = os.path.join(models_path, model_file)
        
        # 根据模型名称创建相应的模型实例
        if '1' in model_name:
            # model = MYLSTMModel(input_size=len(feature_cols))
            model = torch.load('./best_model/all/1cnn-bilstm.pth').to(device)
        elif '2' in model_name:
            # model = ImprovedTransformer(input_size=len(feature_cols))
            model = torch.load('./best_model/all/2my-transformer.pth').to(device)
        elif '3' in model_name:
            model = PureCNN(input_size=len(feature_cols), seq_len=seq_length)
            model.load_state_dict(torch.load('./best_model/all/3cnn.pth'))
            model.to(device)
        elif '4' in model_name:
            model = PureLSTM(input_size=len(feature_cols))
            model.load_state_dict(torch.load('./best_model/all/4lstm.pth'))
            model.to(device)
        elif '5' in model_name:
            model = PureTransformer(input_size=len(feature_cols))
            model.load_state_dict(torch.load('./best_model/all/5transformer.pth'))
            model.to(device)
        
        
        
        # 评估模型
        metrics, predictions, _ = evaluate_model(model, test_loader, criterion, target_scaler, device)
        all_predictions[model_name] = predictions
        all_metrics[model_name] = metrics
        
        # 绘制预测值
        plt.plot(predictions, label=f'{model_name} (RMSE: {metrics["rmse"]:.2f})', alpha=0.7)
    
    plt.title('模型预测结果对比')
    plt.xlabel('时间步')
    plt.ylabel('计数')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(models_path, 'models_comparison.png'), bbox_inches='tight')
    plt.close()
    
    return all_metrics, all_predictions, actuals

def main():
    # 配置参数
    data_path = './data/china_bike_data_2022.csv'
    models_path = './best_model/all'  # 修改为实际的模型保存路径
    seq_length = 12
    batch_size = 64
    
    # 加载和预处理数据
    test_data, feature_scaler, target_scaler = load_and_preprocess_data(data_path)
    
    # 创建测试数据加载器
    x_test, y_test = create_sequences(test_data, feature_cols, 'count', seq_length)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    # 加载模型并进行对比
    print("开始模型对比分析...")
    all_metrics, all_predictions, actuals = compare_models(
        models_path, test_loader, target_scaler, seq_length, device
    )
    
    # 打印对比结果
    print("\n模型对比结果:")
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"WAPE: {metrics['wape']:.2f}%")
        print(f"R²: {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()