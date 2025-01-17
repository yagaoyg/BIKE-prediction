# 共享自行车用量预测
使用深度学习模型对共享单车用量进行预测。
by yagao

## 数据来源
[纽约共享自行车数据集](https://github.com/toddwschneider/nyc-citibike-data)

## 均方根误差公式
\[ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} \]

其中：
- \( n \) 是观测值的数量。
- \( y_i \) 是第 \( i \) 个实际观测值。
- \( \hat{y}_i \) 是第 \( i \) 个预测值。