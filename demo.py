#importing the necessary libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pylab import rcParams
# import tensorflow as tf
from tensorflow import keras
sns.set_style("darkgrid")

df = pd.read_csv('./nyc-citibike-data/data/daily_citi_bike_trip_counts_and_weather.csv')
print(df.head())