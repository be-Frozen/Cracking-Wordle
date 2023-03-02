import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from scipy.stats import goodness_of_fit

# 加载数据
df = pd.read_csv('Problem_C_Data_Wordle.CSV', usecols=['Date', 'Number of reported results'],nrows=300)

# 将 ds 列设为日期时间格式
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Number of reported results': 'y'})
#print(df)
# 将数据集分为训练集和测试集
train_size = int(len(df) * 0.8)
train_df, test_df = df[-train_size:], df[:300-train_size]
#print(train_df)
#print(test_df)
# 训练模型
m = Prophet()
m.fit(train_df)

# 预测
future = m.make_future_dataframe(periods=len(test_df))
forecast = m.predict(future)

# 取出测试集的预测值
test_forecast = forecast.iloc[-len(test_df):]
#print(test_forecast)


# 进行预测
# 计算测试集的拟合优度
#r2 = r2_score(test_df['y'].tolist(), test_forecast['yhat'].tolist())
a=test_df['y'].tolist()
b=test_forecast['yhat'].tolist()
print(a)
a.reverse()
print(a)
print(b)

from sklearn.metrics import mean_squared_error

# 对测试集进行预测

# 计算均方误差
mse = mean_squared_error(a, b)
print('MSE: ', mse)









# 显示测试集预测值和真实值
test_df = test_df.set_index('ds')
test_forecast = test_forecast.set_index('ds')

plt.plot(test_df, label='Actual')
plt.plot(test_forecast['yhat'], label='Predicted')
plt.legend()
plt.show()
