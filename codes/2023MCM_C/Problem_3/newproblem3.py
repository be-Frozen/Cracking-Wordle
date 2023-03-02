import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import radviz
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('Problem_C_Data_Wordle_hello.csv')
X = data[['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']]

# 对数据进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 假设 X 是一个七维数据，使用 PCA 将其降至三维
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 对降维后的数据进行聚类
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_pca)

# 输出每个类别的中心点
centroids = kmeans.cluster_centers_
print("Cluster centroids:")
print(centroids)

# 输出每个数据点在降维后的空间中的坐标
print("3D vectors after dimensionality reduction:")
print(X_pca)

# 计算每个数据点到每个类别中心的距离
distances = kmeans.transform(X_pca)

# 将每个数据点分类到距离最近的中心所对应的类别
labels = np.argmin(distances, axis=1)

# 输出每个数据点与中心的距离及所属类别
print("Distances to centroids and labels:")
for i in range(len(X)):
    print(f"Data point {i}:")
    for j in range(kmeans.n_clusters):
        print(f"Distance to centroid {j}: {distances[i,j]}")
    print(f"Assigned label: {labels[i]}")

# 可视化聚类结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=kmeans.labels_, s=3)
ax.legend(*scatter.legend_elements())
plt.show()
