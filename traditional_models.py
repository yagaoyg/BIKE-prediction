import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("darkgrid")

def load_and_preprocess_data(data_path):
    """加载和预处理数据"""
    df = pd.read_csv(data_path, index_col=['datetime'])
    
    feature_cols = [
        'season', 'year', 'month', 'day', 'dow', 'hour',
        'holiday', 'workingday', 'weather', 'temp',
        'atemp', 'humidity', 'windspeed','regrate'
    ]
    
    train_size = int(len(df) * 0.6)
    test_size = int(len(df) * 0.7)
    
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:test_size]
    
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    train_features = feature_scaler.fit_transform(train_data[feature_cols])
    test_features = feature_scaler.transform(test_data[feature_cols])
    
    train_target = target_scaler.fit_transform(train_data[['count']])
    test_target = target_scaler.transform(test_data[['count']])
    
    return (train_data, test_data, train_features, test_features, 
            train_target, test_target, feature_scaler, target_scaler)

def moving_average(data, window=7):
    """简单移动平均预测"""
    ma = data['count'].rolling(window=window).mean()
    return ma.iloc[window-1:].values

def exponential_smoothing(data, alpha=0.3):
    """指数平滑预测"""
    result = [data['count'].iloc[0]]
    for n in range(1, len(data)):
        result.append(alpha * data['count'].iloc[n] + (1 - alpha) * result[n-1])
    return np.array(result)

def holt_winters(data, season_length=7):
    """Holt-Winters 三重指数平滑预测"""
    y = data['count'].values
    result = []
    
    # 初始化参数
    alpha = 0.3  # 平滑系数
    beta = 0.1   # 趋势系数
    gamma = 0.1  # 季节性系数
    
    # 初始化水平、趋势和季节性因子
    level = y[0]
    trend = (y[season_length] - y[0]) / season_length
    seasonals = y[:season_length] / level
    
    for i in range(len(y)):
        if i >= season_length:
            value = (level + trend) * seasonals[i % season_length]
        else:
            value = y[i]
        
        if i + season_length < len(y):
            level_prev = level
            level = alpha * (y[i] / seasonals[i % season_length]) + (1 - alpha) * (level + trend)
            trend = beta * (level - level_prev) + (1 - beta) * trend
            seasonals[i % season_length] = gamma * (y[i] / level) + (1 - gamma) * seasonals[i % season_length]
        
        result.append(value)
    
    return np.array(result)

def train_linear_regression(train_features, train_target, test_features):
    """训练多元线性回归模型"""
    model = LinearRegression()
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)
    return predictions

def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} 评估结果:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"WAPE: {wape:.2f}%")
    print(f"R²: {r2:.4f}")
    
    return rmse, mape, wape, r2

def plot_predictions(y_true, y_pred, metrics, model_name, save_path):
    """绘制预测结果"""
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, marker='.', label="True Values", alpha=0.7)
    plt.plot(y_pred, marker='.', label="Predictions", alpha=0.7)
    
    rmse, mape, wape, r2 = metrics
    plt.title(f'{model_name} Predictions\nRMSE: {rmse:.2f}, MAPE: {mape:.2f}%, WAPE: {wape:.2f}%, R²: {r2:.4f}')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def main():
    # 加载数据
    data_path = './data/china_bike_data_2022.csv'
    (train_data, test_data, train_features, test_features, 
     train_target, test_target, feature_scaler, target_scaler) = load_and_preprocess_data(data_path)
    
    # 创建保存结果的目录
    save_dir = f'./results/traditional_models/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练和评估不同模型
    results = {}
    
    # 1. 移动平均法
    ma_pred = moving_average(pd.concat([train_data, test_data]))[-len(test_data):]
    metrics = evaluate_model(test_data['count'].values, ma_pred, 'Moving Average')
    results['Moving Average'] = metrics
    plot_predictions(test_data['count'].values, ma_pred, metrics, 'Moving Average', 
                    os.path.join(save_dir, 'ma_predictions.png'))
    
    # 2. 指数平滑法
    es_pred = exponential_smoothing(pd.concat([train_data, test_data]))[-len(test_data):]
    metrics = evaluate_model(test_data['count'].values, es_pred, 'Exponential Smoothing')
    results['Exponential Smoothing'] = metrics
    plot_predictions(test_data['count'].values, es_pred, metrics, 'Exponential Smoothing',
                    os.path.join(save_dir, 'es_predictions.png'))
    
    # 3. Holt-Winters
    hw_pred = holt_winters(pd.concat([train_data, test_data]))[-len(test_data):]
    metrics = evaluate_model(test_data['count'].values, hw_pred, 'Holt-Winters')
    results['Holt-Winters'] = metrics
    plot_predictions(test_data['count'].values, hw_pred, metrics, 'Holt-Winters',
                    os.path.join(save_dir, 'hw_predictions.png'))
    
    # 4. 多元线性回归
    lr_pred = train_linear_regression(train_features, train_target, test_features)
    lr_pred = target_scaler.inverse_transform(lr_pred).flatten()
    metrics = evaluate_model(test_data['count'].values, lr_pred, 'Linear Regression')
    results['Linear Regression'] = metrics
    plot_predictions(test_data['count'].values, lr_pred, metrics, 'Linear Regression',
                    os.path.join(save_dir, 'lr_predictions.png'))
    
    # 保存评估结果
    results_df = pd.DataFrame(results, 
                            index=['RMSE', 'MAPE', 'WAPE', 'R2']).round(4)
    results_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'))
    print("\n模型对比结果:")
    print(results_df)

if __name__ == "__main__":
    main()