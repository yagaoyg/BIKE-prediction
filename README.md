# 共享自行车用量预测
使用深度学习模型对共享单车用量进行预测。
by yagao

## 数据来源

DATA:
London Bike Sharing data is available in Kaggle.
A bicycle-sharing system, public bicycle scheme, or public bike share (PBS) scheme,
is a service in which bicycles are made available for shared use to individuals on a short term basis for a price or free.

Our goal is to predict the number of future bike shares given the historical data of London bike shares. 

What features do we have?

timestamp - timestamp field for grouping the data

cnt - the count of a new bike shares

t1 - real temperature in C

t2 - temperature in C “feels like”

hum - humidity in percentage

wind_speed - wind speed in km/h

weather_code - category of the weather

is_holiday - boolean field - 1 holiday / 0 non holiday

is_weekend - boolean field - 1 if the day is weekend

season - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.

We need  to predict the count of future bike shares.
