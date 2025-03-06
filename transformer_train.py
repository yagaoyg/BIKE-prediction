# 自行车使用量预测Transformer完整实现
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from keras.api.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.api.models import Model, load_model
from keras.api.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.api.optimizers import Adam
from sklearn.metrics import mean_squared_error

# 配置设置
sns.set_style("darkgrid")
tf.keras.utils.set_random_seed(42)  # 固定随机种子

# 硬件检测
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU可用 - 检测到 {len(gpus)} 块显卡")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("GPU不可用，使用CPU")

check_gpu()

# 数据加载与预处理
def load_and_preprocess(data_path='./data/daily_citi_bike_trip_counts_and_weather.csv'):
    raw_data = pd.read_csv(
        data_path,
        parse_dates=['date'],
        index_col='date'
    )
    
    # 特征列定义
    feature_cols = [
        'precipitation', 'snow_depth', 'snowfall', 'max_t', 'min_t',
        'average_wind_speed', 'dow', 'year', 'month', 'holiday',
        'weekday', 'weekday_non_holiday', 'dt', 'season'
    ]
    target_col = 'trips'
    
    # 数据集划分
    train_size = int(len(raw_data) * 0.9)
    train_data = raw_data.iloc[:train_size]
    test_data = raw_data.iloc[train_size:]
    
    # 特征标准化
    feature_scaler = RobustScaler().fit(train_data[feature_cols])
    train_data.loc[:, feature_cols] = feature_scaler.transform(train_data[feature_cols])
    test_data.loc[:, feature_cols] = feature_scaler.transform(test_data[feature_cols])
    
    # 目标值标准化
    target_scaler = RobustScaler().fit(train_data[[target_col]])
    train_data[target_col] = target_scaler.transform(train_data[[target_col]])
    test_data[target_col] = target_scaler.transform(test_data[[target_col]])
    
    return train_data, test_data, feature_cols, target_col, target_scaler

# 时间序列数据集生成
def create_sequences(data, feature_cols, target_col, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[feature_cols].iloc[i:i+time_steps].values)
        y.append(data[target_col].iloc[i+time_steps])
    return np.array(X), np.array(y)

# Transformer模型构建
def build_transformer_model(input_shape, num_heads=4, ff_dim=256, dropout=0.1):
    inputs = Input(shape=input_shape)
    
    # 位置编码层
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embedding = tf.keras.layers.Embedding(
        input_dim=input_shape[0], 
        output_dim=input_shape[1]
    )(positions)
    x = inputs + position_embedding
    
    # Transformer编码块
    for _ in range(2):
        # 自注意力机制
        attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=input_shape[1]//num_heads
        )(x, x)
        attn_output = Dropout(dropout)(attn_output)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # 前馈网络
        ffn = Dense(ff_dim, activation="gelu")(x)
        ffn = Dense(input_shape[1])(ffn)
        ffn = Dropout(dropout)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # 输出层
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    return model

# 主程序流程
def main():
    # 数据准备
    train_data, test_data, feature_cols, target_col, target_scaler = load_and_preprocess()
    
    # 生成序列数据
    TIME_STEPS = 1  # 使用7天历史数据进行预测
    X_train, y_train = create_sequences(train_data, feature_cols, target_col, TIME_STEPS)
    X_test, y_test = create_sequences(test_data, feature_cols, target_col, TIME_STEPS)
    
    # 模型配置
    model = build_transformer_model(
        input_shape=(TIME_STEPS, len(feature_cols)),
        num_heads=4,
        ff_dim=256,
        dropout=0.2
    )
    
    # 模型编译
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    
    # 回调函数
    base_path = f"./model/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    os.makedirs(base_path, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(base_path, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )
    
    # 模型训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=12000,
        batch_size=128,
        callbacks=[
          # checkpoint, 
          lr_scheduler
          ],
        verbose=1
    )
    
    # 加载最佳模型
    # best_model = load_model(os.path.join(base_path, 'best_model.keras'))
    
    # 预测与评估
    # y_pred = best_model.predict(X_test)
    y_pred = model.predict(X_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f'测试集RMSE: {rmse:.2f}')
    
    min_val_loss = round(min(history.history['val_loss']),6)
    
    # 绘制训练损失和验证损失的变化曲线
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'],label='train loss')
    plt.plot(history.history['val_loss'],label='vall loss')
    plt.legend()
    plt.savefig(base_path +'/loss ' + str(min_val_loss) + '.png')
    plt.show()
    
    # 可视化结果
    plt.figure(figsize=(12, 4))
    plt.plot(y_test_inv, label='true', alpha=0.7)
    plt.plot(y_pred_inv, label='pred', alpha=0.7)
    plt.title(f'Transformer预测效果 (RMSE={rmse:.2f})')
    plt.legend()
    plt.savefig(os.path.join(base_path, 'prediction_comparison.png'))
    plt.show()
    
    # 保存完整模型
    # best_model.save(os.path.join(base_path, 'final_transformer_model.keras'))
    model.save(os.path.join(base_path, 'final_transformer_model.keras'))

if __name__ == "__main__":
    main()