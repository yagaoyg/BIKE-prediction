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

# 设置全局字体样式和大小
sns.set_style("darkgrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 18  # 增大基础字体大小
plt.rcParams['axes.titlesize'] = 20  # 增大标题字体大小
plt.rcParams['axes.labelsize'] = 18  # 增大轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16  # 增大x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16  # 增大y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 18  # 增大图例字体大小

data_name = 'daily_citi_bike_trip_counts_and_weather'
# data_name = 'china_bike_data_2022'

# 引入数据集
data = pd.read_csv('./data/' + data_name + '.csv',
                   parse_dates=['date'],
                   index_col=['date'],
                #    usecols=['date',
                #             'trips',
                #             'precipitation',
                #             'snowfall',
                #             'max_t',
                #             'min_t',
                #             'average_wind_speed',
                #             'dow',
                #             'holiday',
                #             'stations_in_service',
                #             'weekday',
                #             'weekday_non_holiday',
                #             'month',
                #             'dt',
                #             'day',
                #             'year']
                   )

# data = pd.read_csv('./data/' + data_name + '.csv',
#                    parse_dates=['datetime'],
#                    index_col=['datetime'])

# 以日期为x轴，骑行次数为y轴，绘制折线图
plt.figure(figsize=(12,5))
sns.lineplot(data=data, x=data.index, y=data.trips)
plt.xlabel('日期')  # 设置X轴标签
plt.ylabel('骑行次数')  # 设置Y轴标签
plt.savefig('./output/trips of day.png')
plt.show()

# 以日期为x轴，骑行次数为y轴，绘制折线图
# plt.figure(figsize=(15,5))
# sns.lineplot(data=data, x=data.index, y=data['count'])
# plt.show()

# print(data.head())

# 以月份为x轴，骑行次数为y轴，绘制折线图
data_month = data.resample('M').sum()
plt.figure(figsize=(12,5))
sns.lineplot(data=data_month, x=data_month.index, y=data_month.trips, marker='o')
plt.xlabel('月份')  # 设置X轴标签
plt.ylabel('骑行次数')  # 设置Y轴标签
plt.savefig('./output/trips of month.png')
plt.show()

# 数值型特征的统计分析
# numerical_features1 = ['precipitation', 'snowfall']
# numerical_features2 = ['snow_depth','max_t']
# numerical_features3 = ['min_t', 'dt']
# numerical_features4 = ['average_wind_speed','stations_in_service']

# 绘制数值型特征的分布直方图
# plt.figure(figsize=(12, 6))
# for i, feature in enumerate(numerical_features1, 1):
#     plt.subplot(1, 2, i)
#     plt.ylabel('次数')  # 设置Y轴标签
#     sns.histplot(data=data[feature], kde=True)
#     # plt.title(f'{feature}')
# plt.tight_layout()
# plt.savefig('./output/' + data_name + '_numerical_features_distribution1.png')
# plt.show()

# plt.figure(figsize=(12, 6))
# for i, feature in enumerate(numerical_features2, 1):
#     plt.subplot(1, 2, i)
#     plt.ylabel('次数')  # 设置Y轴标签
#     sns.histplot(data=data[feature], kde=True)
#     # plt.title(f'{feature}')
# plt.tight_layout()
# plt.savefig('./output/' + data_name + '_numerical_features_distribution2.png')
# plt.show()

# plt.figure(figsize=(12, 6))
# for i, feature in enumerate(numerical_features3, 1):
#     plt.subplot(1, 2, i)
#     plt.ylabel('次数')  # 设置Y轴标签
#     sns.histplot(data=data[feature], kde=True)
#     # plt.title(f'{feature}')
# plt.tight_layout()
# plt.savefig('./output/' + data_name + '_numerical_features_distribution3.png')
# plt.show()

# plt.figure(figsize=(12, 6))
# for i, feature in enumerate(numerical_features4, 1):
#     plt.subplot(1, 2, i)
#     plt.ylabel('次数')  # 设置Y轴标签
#     sns.histplot(data=data[feature], kde=True)
#     # plt.title(f'{feature}')
# plt.tight_layout()
# plt.savefig('./output/' + data_name + '_numerical_features_distribution4.png')
# plt.show()

# 共享自行车用量的分布
# numerical_features = 'trips'
# stats_df = data[numerical_features].agg(['mean', 'std', 'skew', 'kurt']).round(2)
# print("\n数值型特征的统计分析：")
# print(stats_df)

# 绘制自行车用量的分布直方图
# plt.figure(figsize=(6, 6))
# sns.histplot(data=data[numerical_features], kde=True)
# plt.title(f'{numerical_features}')
# plt.tight_layout()
# plt.savefig('./output/' + data_name + 'trips_distribution.png')
# plt.show()

# 服务中的站点数量
# plt.figure(figsize=(10,5))
# sns.lineplot(data=data, x=data.index, y='stations_in_service')
# plt.xlabel('日期')  # 设置X轴标签
# plt.ylabel('站点数量')  # 设置Y轴标签
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
# plt.figure(figsize=(17, 17))
# sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title("heatmap")
# plt.savefig('./output/heatmap.png')
# plt.show()

# 分析所有特征与骑行次数的关系2
# corr = data[['precipitation','snowfall','max_t','min_t','average_wind_speed','dow','holiday','stations_in_service','weekday','weekday_non_holiday','month','dt','day','year','trips']].corr()
# plt.figure(figsize=(14, 14))
# sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title("heatmap")
# plt.savefig('./output/heatmap2.png')
# plt.show()

# 按时间显示运营中的站点数量
# plt.figure(figsize=(15,5))
# sns.lineplot(data=data, x=data.index, y='stations_in_service')
# plt.show()

# 按季节显示骑行次数
# seasonal_data = data.groupby('season').mean()
# plt.figure(figsize=(10,5))
# ax = sns.barplot(data=seasonal_data, x='season', y=seasonal_data.trips,palette='viridis')
# ax.set_xlabel('季节')  # 设置X轴标签
# ax.set_ylabel('平均骑行次数')  # 设置Y轴标签
# # 在柱状图上添加数值标注
# for i in ax.containers:
#     ax.bar_label(i, fmt='%.0f')
# plt.savefig('./output/season trips.png')
# plt.show()

# 按星期显示骑行次数
# dow_data = data.groupby('dow').mean()
# plt.figure(figsize=(10,5))
# ax = sns.barplot(data=dow_data, x='dow', y=dow_data.trips, palette='viridis')
# ax.set_xlabel('星期')  # 设置X轴标签
# ax.set_ylabel('平均骑行次数')  # 设置Y轴标签
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

# 分析假期相关特征的分布
plt.figure(figsize=(21, 7))

# # 计算每个类别的平均值
holiday_means = data.groupby('holiday')['trips'].mean().round(0)
weekday_means = data.groupby('weekday')['trips'].mean().round(0)
weekday_non_holiday_means = data.groupby('weekday_non_holiday')['trips'].mean().round(0)

# holiday分布及其与trips的关系
plt.subplot(131)
sns.boxplot(x='holiday', y='trips', data=data)
plt.xlabel('是否假期')  # 设置X轴标签
plt.ylabel('骑行次数')  # 设置Y轴标签
# plt.title('Holiday vs Trips')
# 添加均值标注
for i, value in enumerate(holiday_means):
    y_pos = data[data['holiday'] == i]['trips'].max() + 1000
    plt.text(i, y_pos, f'平均值:\n{int(value)}', 
             horizontalalignment='center', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             rotation=0)  # 设置水平方向

# weekday分布及其与trips的关系
plt.subplot(132)
sns.boxplot(x='weekday', y='trips', data=data)
plt.xlabel('是否工作日')  # 设置X轴标签
plt.ylabel('骑行次数')  # 设置Y轴标签
# plt.title('Weekday vs Trips')
# 添加均值标注
for i, value in enumerate(weekday_means):
    y_pos = data[data['weekday'] == i]['trips'].max() + 1000
    plt.text(i, y_pos, f'平均值:\n{int(value)}', 
             horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             rotation=0)  # 设置水平方向

# weekday_non_holiday分布及其与trips的关系
plt.subplot(133)
sns.boxplot(x='weekday_non_holiday', y='trips', data=data)
plt.xlabel('是否纯工作日')  # 设置X轴标签
plt.ylabel('骑行次数')  # 设置Y轴标签
# plt.title('Weekday Non-Holiday vs Trips')
# 添加均值标注
for i, value in enumerate(weekday_non_holiday_means):
    y_pos = data[data['weekday_non_holiday'] == i]['trips'].max() + 1000
    plt.text(i, y_pos, f'平均值:\n{int(value)}', 
             horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             rotation=0)  # 设置水平方向

plt.tight_layout()
plt.savefig('./output/holiday_features_analysis.png')
plt.show()

# 打印各特征的统计信息
# print("\n各特征的频数统计：")
# print("\nHoliday频数：")
# print(data['holiday'].value_counts())
# print("\nWeekday频数：")
# print(data['weekday'].value_counts())
# print("\nWeekday Non-Holiday频数：")
# print(data['weekday_non_holiday'].value_counts())

# 计算均值统计
# print("\n各特征下的骑行次数均值：")
# print("\nHoliday情况下的平均骑行次数：")
# print(data.groupby('holiday')['trips'].mean())
# print("\nWeekday情况下的平均骑行次数：")
# print(data.groupby('weekday')['trips'].mean())
# print("\nWeekday Non-Holiday情况下的平均骑行次数：")
# print(data.groupby('weekday_non_holiday')['trips'].mean())

# 计算特征占比
# total_days = len(data)
# print("\n各特征在总天数中的占比：")

# print("\nHoliday占比：")
# holiday_ratio = (data['holiday'].value_counts() / total_days * 100).round(2)
# print(holiday_ratio.to_string(header=False) + ' %')

# print("\nWeekday占比：")
# weekday_ratio = (data['weekday'].value_counts() / total_days * 100).round(2)
# print(weekday_ratio.to_string(header=False) + ' %')

# print("\nWeekday Non-Holiday占比：")
# weekday_non_holiday_ratio = (data['weekday_non_holiday'].value_counts() / total_days * 100).round(2)
# print(weekday_non_holiday_ratio.to_string(header=False) + ' %')

# 绘制特征占比饼图
plt.figure(figsize=(16, 5))

# Holiday占比饼图
plt.subplot(131)
holiday_counts = data['holiday'].value_counts()
plt.pie(holiday_counts, autopct='%1.1f%%', textprops={'color': 'white'})
plt.title('假期分布')
plt.legend(holiday_counts.index, loc='best')

# Weekday占比饼图
plt.subplot(132)
weekday_counts = data['weekday'].value_counts()
plt.pie(weekday_counts, labels=weekday_counts.index, autopct='%1.1f%%', textprops={'color': 'white'})
plt.title('工作日分布')
plt.legend(weekday_counts.index, loc='best')

# Weekday Non-Holiday占比饼图
plt.subplot(133)
weekday_non_holiday_counts = data['weekday_non_holiday'].value_counts()
plt.pie(weekday_non_holiday_counts, labels=weekday_non_holiday_counts.index, autopct='%1.1f%%', textprops={'color': 'white'})
plt.title('纯工作日分布')
plt.legend(weekday_non_holiday_counts.index, loc='best')

plt.tight_layout()
plt.savefig('./output/holiday_features_pie_charts.png')
plt.show()

# 分析分类变量的非线性关系
plt.figure(figsize=(12, 12))

# dow的箱线图分析
plt.subplot(221)
sns.boxplot(x='dow', y='trips', data=data)
plt.xlabel('星期')  # 设置X轴标签
plt.ylabel('骑行次数')  # 设置Y轴标签
plt.title('星期与自行车用量')

# holiday的箱线图分析
plt.subplot(222)
sns.boxplot(x='holiday', y='trips', data=data)
plt.xlabel('假期')  # 设置X轴标签
plt.ylabel('骑行次数')  # 设置Y轴标签
plt.title('假期与自行车用量')

# 分析连续变量的非线性关系
plt.subplot(223)
sns.regplot(x='day', y='trips', data=data, scatter_kws={'alpha':0.5}, 
            order=2, line_kws={'color': 'red'})
plt.xlabel('月中日期')  # 设置X轴标签
plt.ylabel('骑行次数')  # 设置Y轴标签
plt.title('月中日期与自行车用量')

plt.subplot(224)
sns.regplot(x='year', y='trips', data=data, scatter_kws={'alpha':0.5}, 
            order=2, line_kws={'color': 'red'})
plt.xlabel('年份')  # 设置X轴标签
plt.ylabel('骑行次数')  # 设置Y轴标签
plt.title('年份与自行车用量')

plt.tight_layout()
plt.savefig('./output/nonlinear_analysis.png')
plt.show()

# # 计算多项式特征与trips的相关性
# poly = PolynomialFeatures(degree=2, include_bias=False)
# day_poly = poly.fit_transform(data[['day']])
# year_poly = poly.fit_transform(data[['year']])

# # 多项式回归
# day_reg = LinearRegression().fit(day_poly, data['trips'])
# year_reg = LinearRegression().fit(year_poly, data['trips'])

# print("\n多项式回归R²值：")
# print(f"Day polynomial R²: {day_reg.score(day_poly, data['trips']):.4f}")
# print(f"Year polynomial R²: {year_reg.score(year_poly, data['trips']):.4f}")

# 计算互信息分数
# print("\n互信息分数（反映非线性相关性）：")
# print("互信息强度分级参考：")
# print("0-0.1: 极弱相关或无相关")
# print("0.1-0.3: 弱相关")
# print("0.3-0.5: 中等相关")
# print("0.5-0.8: 强相关")
# print("0.8-1.0: 极强相关")
# print("\n各特征的互信息分数：")
# print(f"Dow MI: {mutual_info_score(data['dow'], pd.qcut(data['trips'], q=10, labels=False, duplicates='drop')):.4f}")
# print(f"Holiday MI: {mutual_info_score(data['holiday'], pd.qcut(data['trips'], q=10, labels=False, duplicates='drop')):.4f}")
# print(f"Day MI: {mutual_info_score(pd.qcut(data['day'], q=10, labels=False, duplicates='drop'), pd.qcut(data['trips'], q=10, labels=False, duplicates='drop')):.4f}")
# print(f"Year MI: {mutual_info_score(pd.qcut(data['year'], q=10, labels=False, duplicates='drop'), pd.qcut(data['trips'], q=10, labels=False, duplicates='drop')):.4f}")

# # 可视化互信息分数
plt.figure(figsize=(10, 6))
mi_scores = {
    'Dow': mutual_info_score(data['dow'], pd.qcut(data['trips'], q=10, labels=False, duplicates='drop')),
    'Holiday': mutual_info_score(data['holiday'], pd.qcut(data['trips'], q=10, labels=False, duplicates='drop')),
    'Day': mutual_info_score(pd.qcut(data['day'], q=10, labels=False, duplicates='drop'), pd.qcut(data['trips'], q=10, labels=False, duplicates='drop')),
    'Year': mutual_info_score(pd.qcut(data['year'], q=10, labels=False, duplicates='drop'), pd.qcut(data['trips'], q=10, labels=False, duplicates='drop'))
}

# 创建条形图
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(mi_scores)), list(mi_scores.values()), color='skyblue')
plt.xticks(range(len(mi_scores)), list(mi_scores.keys()))
plt.title('互信息分析')
plt.ylabel('互信息分数')

# 在柱状图上添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

# 添加参考线
plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.3, label='极弱相关阈值(0.1)')
plt.axhline(y=0.3, color='g', linestyle='--', alpha=0.3, label='弱相关阈值(0.3)')
plt.axhline(y=0.5, color='b', linestyle='--', alpha=0.3, label='中等相关阈值(0.5)')
plt.axhline(y=0.8, color='purple', linestyle='--', alpha=0.3, label='强相关阈值(0.8)')

plt.legend()
plt.tight_layout()
plt.savefig('./output/mutual_information_scores.png')
plt.show()

# 分析所有特征的互信息分数
# def calculate_mutual_info(data, feature, target='trips'):
#     if data[feature].dtype in ['int64', 'float64']:
#         # 对连续型变量进行分箱处理
#         feature_binned = pd.qcut(data[feature], q=10, labels=False, duplicates='drop')
#     else:
#         feature_binned = data[feature]
    
#     target_binned = pd.qcut(data[target], q=10, labels=False, duplicates='drop')
#     return mutual_info_score(feature_binned, target_binned)

# # 计算所有特征的互信息分数
# mi_scores = {}
# for feature in data.columns:
#     if feature != 'trips':
#         mi_scores[feature] = calculate_mutual_info(data, feature)

# # 将互信息分数转换为DataFrame并排序
# mi_df = pd.DataFrame(mi_scores.items(), columns=['Feature', 'MI_Score'])
# mi_df = mi_df.sort_values('MI_Score', ascending=False)

# # 绘制所有特征的互信息分数条形图
# plt.figure(figsize=(12, 6))
# bars = plt.bar(range(len(mi_df)), mi_df['MI_Score'], color='skyblue')
# plt.xticks(range(len(mi_df)), mi_df['Feature'], rotation=45, ha='right')
# plt.title('所有特征与骑行次数的互信息分数')
# plt.ylabel('互信息分数')

# # 在柱状图上添加具体数值
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,
#              f'{height:.4f}',
#              ha='center', va='bottom')

# # 添加参考线
# plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.3, label='极弱相关(0.1)')
# plt.axhline(y=0.3, color='g', linestyle='--', alpha=0.3, label='弱相关(0.3)')
# plt.axhline(y=0.5, color='b', linestyle='--', alpha=0.3, label='中等相关(0.5)')
# plt.axhline(y=0.8, color='purple', linestyle='--', alpha=0.3, label='强相关(0.8)')

# plt.legend()
# plt.tight_layout()
# plt.savefig('./output/all_features_mutual_information.png')
# plt.show()

# # 打印互信息分数表格
# print("\n所有特征与骑行次数的互信息分数（按分数降序排列）：")
# print(mi_df.to_string(index=False))