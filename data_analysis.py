# 用于分析数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
    
data_name = 'daily_citi_bike_trip_counts_and_weather'

# 引入数据集
data = pd.read_csv('./data/' + data_name + '.csv',
                   parse_dates=['date'],
                   index_col=['date'],
                  #  usecols=['date',
                  #           'trips',
                  #           'precipitation',
                  #           'snowfall',
                  #           'max_t',
                  #           'min_t',
                  #           'average_wind_speed',
                  #           'dow',
                  #           'holiday',
                  #           'stations_in_service',
                  #           'weekday',
                  #           'weekday_non_holiday',
                  #           'month',
                  #           'dt',
                  #           'day',
                  #           'year']
                   )

# 以日期为x轴，骑行次数为y轴，绘制折线图
plt.figure(figsize=(15,5))
sns.lineplot(data=data, x=data.index, y=data.trips)
plt.show()

# 以月份为x轴，骑行次数为y轴，绘制折线图
data_month = data.resample('M').sum()
plt.figure(figsize=(15,5))
sns.lineplot(data=data_month, x=data_month.index, y=data_month.trips)
plt.show()

# 降雨量
# plt.figure(figsize=(15,5))
# ax1 = plt.gca()  # 主坐标轴
# ax2 = ax1.twinx()  # 共享x轴的次坐标轴

# # 绘制骑行次数（主Y轴）
# sns.lineplot(data=data, x=data.index, y='trips', ax=ax1, label='Trips')
# # 绘制降水量（次Y轴）
# sns.lineplot(data=data, x=data.index, y='precipitation', ax=ax2, color='orange', label='Precipitation')

# ax1.set_ylabel('Trips (次)')
# ax2.set_ylabel('Precipitation (mm)')
# plt.title('骑行次数与降水量时序对比')
# plt.show()

# 分析降雨量和骑行次数的相关性 计算它们之间的皮尔逊相关系数
# correlation = data[['trips', 'precipitation']].corr()
# print(correlation)

# 热力图可视化 绘制相关系数
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title("变量间相关系数热力图")
# plt.show()


# 站点数量与骑行次数的关系
# plt.figure(figsize=(15,5))
# ax1 = plt.gca()  # 主坐标轴
# ax2 = ax1.twinx()  # 共享x轴的次坐标轴

# # 绘制骑行次数（主Y轴）
# sns.lineplot(data=data, x=data.index, y='trips', ax=ax1, label='Trips')
# # 绘制降水量（次Y轴）
# sns.lineplot(data=data, x=data.index, y='stations_in_service', ax=ax2, color='orange', label='Precipitation')

# ax1.set_ylabel('Trips (次)')
# ax2.set_ylabel('开放的站点数量')
# plt.show()


# 分析所有特征与骑行次数的关系
# corr = data.corr()
# plt.figure(figsize=(13, 13))
# sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title("变量间相关系数热力图")
# plt.show()

# 按时间显示运营中的站点数量
# plt.figure(figsize=(15,5))
# sns.lineplot(data=data, x=data.index, y='stations_in_service')
# plt.show()

# 按季节显示骑行次数
# seasonal_data = data.groupby('season').sum()
# plt.figure(figsize=(8,4))
# sns.barplot(data=seasonal_data, x='season', y=seasonal_data.trips,palette='viridis')
# plt.show()

# 按星期显示骑行次数
# dow_data = data.groupby('dow').sum()
# plt.figure(figsize=(10,4))
# sns.barplot(data=dow_data, x='dow', y=dow_data.trips,palette='viridis')
# plt.show()

