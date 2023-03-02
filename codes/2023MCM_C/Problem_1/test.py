import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt


df = pd.read_csv('Problem_C_Data_Wordle.CSV', usecols=['Date', 'Number of reported results'], nrows=300)

df = df.rename(columns={'Date': 'ds', 'Number of reported results':'y'})

m = Prophet()

m.fit(df)

future = m.make_future_dataframe(periods=100)
forecast = m.predict(future)

# 将 ds 列设为索引
forecast1 = forecast.set_index('ds')
# 查询特定日期的预测值
prediction = forecast1.loc['2023-03-01']
# 显示预测值
print('2023-03-01 的预测值为：', prediction['yhat'])
print('2023-03-01 的置信区间为：', prediction[['yhat_lower', 'yhat_upper']])

fig1 = m.plot(forecast)
#fig2 = m.plot_components(forecast)

plt.title('Number of Reported Results (Prophet)')
plt.xlabel('Date')
plt.ylabel('Number of reported results')
plt.xticks(['2022-03-01', '2022-05-01', '2022-07-01', '2022-09-01', '2022-11-01', '2023-01-01', '2023-03-01'])
plt.show()


