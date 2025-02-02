# 用于测试训练数据指标保存的代码
import pandas as pd
from datetime import datetime

train_df = pd.read_excel('train.xlsx')

start_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())

print(train_df.head())

end_time = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())

# new_data = {'date':TIMESTAMP,'train_percentage':0.9,'time_steps':1,'l1':80,'d1':0.4,'l2':64,'d2':0.3,'epochs':1.00,'batch_size':32,'rmse_lstm':5555}
new_df = pd.DataFrame([[start_time,end_time,0.9,1,80,0.4,64,0.3,1000,32,5555]],columns=['start_time','end_time','train_percentage','time_steps','l1','d1','l2','d2','epochs','batch_size','rmse_lstm'])

# save_data = pd.DataFrame()
save_data = train_df._append(new_df)

print(save_data.head())

save_data.to_excel('train.xlsx',index=False)