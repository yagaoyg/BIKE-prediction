# 用于分析数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mutual_info_score
# from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

sns.set_style("darkgrid")
    
data_name = 'daily_citi_bike_trip_counts_and_weather'
# data_name = 'china_bike_data_2022'

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

# data = pd.read_csv('./data/' + data_name + '.csv',
#                    parse_dates=['datetime'],
#                    index_col=['datetime'])

# 以日期为x轴，骑行次数为y轴，绘制折线图
plt.figure(figsize=(15,5))
sns.lineplot(data=data, x=data.index, y=data.trips)
plt.savefig('./output/trips of day.png')
plt.show()

# 以日期为x轴，骑行次数为y轴，绘制折线图
# plt.figure(figsize=(15,5))
# sns.lineplot(data=data, x=data.index, y=data['count'])
# plt.show()

# print(data.head())

# 以月份为x轴，骑行次数为y轴，绘制折线图
data_month = data.resample('M').sum()
plt.figure(figsize=(15,5))
sns.lineplot(data=data_month, x=data_month.index, y=data_month.trips, marker='o')
plt.savefig('./output/trips of month.png')
plt.show()

# 数值型特征的统计分析
numerical_features = ['precipitation', 'snowfall', 'snow_depth','max_t', 'min_t', 'average_wind_speed','stations_in_service','dt']
stats_df = data[numerical_features].agg(['mean', 'std', 'skew', 'kurt']).round(2)
print("\n数值型特征的统计分析：")
print(stats_df)

# 绘制数值型特征的分布直方图
plt.figure(figsize=(16, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 4, i)
    sns.histplot(data=data[feature], kde=True)
    plt.title(f'{feature}')
plt.tight_layout()
plt.savefig('./output/' + data_name + '_numerical_features_distribution.png')
plt.show()

# 共享自行车用量的分布
numerical_features = 'trips'
stats_df = data[numerical_features].agg(['mean', 'std', 'skew', 'kurt']).round(2)
print("\n数值型特征的统计分析：")
print(stats_df)

# 绘制数值型特征的分布直方图
plt.figure(figsize=(6, 6))
sns.histplot(data=data[feature], kde=True)
plt.title(f'{feature}')
plt.tight_layout()
plt.savefig('./output/' + data_name + 'trips_distribution.png')
plt.show()

# 服务中的站点数量
# plt.figure(figsize=(10,4))
# sns.lineplot(data=data, x=data.index, y='stations_in_service')
# plt.savefig('./output/stations_in_service.png')
# plt.show()

# 分析stations_in_service前700条数据的波动性
# sample_data = data['stations_in_service'].head(700)

# 计算基本统计量
# volatility_stats = {
#     '标准差': sample_data.std(),
#     '变异系数': sample_data.std() / sample_data.mean(),
#     '最大波动范围': sample_data.max() - sample_data.min()
# }
# print("\nstations_in_service前700条数据的波动性统计：")
# print(pd.Series(volatility_stats).round(2))

# 绘制时序图和移动平均线
# plt.figure(figsize=(15, 6))
# plt.plot(sample_data.index, sample_data, label='ori data')
# plt.legend()
# plt.show()

# 验证有无前31条数据的分布一致性
# full_data = data['stations_in_service']
# data_without_31 = data['stations_in_service'][31:]

# # 进行K-S检验
# ks_statistic, p_value = stats.ks_2samp(full_data, data_without_31)

# print("\n有无前31条数据的K-S检验结果：")
# print(f"KS统计量: {ks_statistic:.4f}")
# print(f"P值: {p_value:.4f}")
# print(f"结论: {'数据分布一致' if p_value > 0.05 else '数据分布不一致'}")

# 绘制经验分布函数对比图
# plt.figure(figsize=(10, 6))
# plt.step(np.sort(full_data), np.arange(1, len(full_data) + 1) / len(full_data),
#          label='full data', where='post')
# plt.step(np.sort(data_without_31), np.arange(1, len(data_without_31) + 1) / len(data_without_31),
#          label='without 31', where='post')
# plt.xlabel('站点服务数量')
# plt.ylabel('累积概率')
# plt.title('K-S检验：去除前31条数据的分布对比')
# plt.legend()
# plt.grid(True)
# plt.show()

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
# plt.title("heatmap")
# plt.show()

# 按时间显示运营中的站点数量
# plt.figure(figsize=(15,5))
# sns.lineplot(data=data, x=data.index, y='stations_in_service')
# plt.show()

# 按季节显示骑行次数
# seasonal_data = data.groupby('season').mean()
# plt.figure(figsize=(8,4))
# ax = sns.barplot(data=seasonal_data, x='season', y=seasonal_data.trips,palette='viridis')
# # 在柱状图上添加数值标注
# for i in ax.containers:
#     ax.bar_label(i, fmt='%.0f')
# plt.savefig('./output/season trips.png')
# plt.show()

# 按星期显示骑行次数
# dow_data = data.groupby('dow').mean()
# plt.figure(figsize=(10,4))
# ax = sns.barplot(data=dow_data, x='dow', y=dow_data.trips, palette='viridis')
# # 在柱状图上添加数值标注
# for i in ax.containers:
#     ax.bar_label(i, fmt='%.0f')
# plt.savefig('./output/dow trips.png')
# plt.show()

# 是否假期与骑行次数的相关性分析
# print("描述统计:")
# print(data.groupby('holiday')['trips'].describe())

# 年份、月中日期与骑行次数的关系
# ====== 关联性强弱判断 ======
# def check_association(feature, target):
#     """快速评估特征与目标变量的关联性"""
#     # 数值型关联
#     # if feature.nunique() > 10:
#     #     corr = stats.pearsonr(feature, target)
#     #     print(f"皮尔逊相关系数: {corr[0]:.2f} (p={corr[1]:.3f})")
#     #     return abs(corr[0])
    
#     # 类别型关联
#     # else:
#     groups = [target[feature == val] for val in feature.unique()]
#     h_stat, p_val = stats.kruskal(*groups)
#     print(f"Kruskal-Wallis检验: H={h_stat:.1f} (p={p_val:.3f})")
#     return h_stat / 800  # 标准化效应量

# print('\n')

# # 评估年份关联性
# print("年份分析:")
# year_strength = check_association(data['year'], data['trips'])

# # 评估日期关联性
# print("\n月中日期分析:")
# day_strength = check_association(data['day'], data['trips'])

# # 评估星期关联性
# print("\n星期分析:")
# weekday_strength = check_association(data['weekday'], data['trips'])

# # 评估假期关联性
# print("\n假期分析:")
# holiday_strength = check_association(data['holiday'], data['trips'])

# # ====== 可视化验证 ======
# plt.figure(figsize=(12,5))

# # 年份趋势
# plt.subplot(1,2,1)
# sns.lineplot(x='year', y='trips', data=data, ci=None, estimator='mean')
# plt.title('nian du yong liang')

# # 日分布
# plt.subplot(1,2,2)
# sns.boxplot(x='day', y='trips', data=data, showfliers=False)
# plt.xticks(rotation=90)
# plt.title('mei ri yong liang')

# plt.tight_layout()
# plt.show()

# # 假期分布
# plt.figure(figsize=(10,5))
# sns.barplot(x='holiday', y='trips', data=data, palette='viridis',estimator='mean')
# plt.title('holiday yong liang')
# plt.show()

# # ====== 决策建议 ======
# thresholds = {
#     '弱关联': 0.1,
#     '极弱关联': 0.05
# }

# print(f"\n决策建议:")
# print(f"年份关联强度: {year_strength:.2f}（{'可用' if year_strength > thresholds['极弱关联'] else '建议舍弃'}）")
# print(f"日期关联强度: {day_strength:.2f}（{'可用' if day_strength > thresholds['极弱关联'] else '建议舍弃'}）")
# print(f"星期关联强度: {weekday_strength:.2f}（{'可用' if weekday_strength > thresholds['极弱关联'] else '建议舍弃'}）")
# print(f"假期关联强度: {holiday_strength:.2f}（{'可用' if holiday_strength > thresholds['极弱关联'] else '建议舍弃'}）")

