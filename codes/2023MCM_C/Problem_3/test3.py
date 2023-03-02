import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import radviz
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np

data=pd.read_csv('Problem_C_Data_Wordle_hello.csv')
X = data[['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']]


from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# 对数据进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X)
# 假设 X 是一个七维数据，使用 PCA 将其降至三维
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
#print(X_pca)
# 对降维后的数据进行聚类
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_pca)

# [0.55119, 9.36539, 27.56553, 33.76631, 22.30818, 8.67281, 3.51006]

new_data = [[2, 11, 34, 32, 15, 6, 1]]
#new_data=X
#new_data=new_data.values.tolist()
# 对新数据进行标准化
scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(new_data)
# 使用之前的PCA对象将新数据转换成3维
new_data_pca = pca.transform(new_data)
#new_data_pca = pca.transform(new_data_scaled)

# 使用之前的聚类模型预测新数据所属的类别
new_data_label = kmeans.predict(new_data_pca)

print(new_data_label)


# 可视化聚类结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=kmeans.labels_)
plt.show()

