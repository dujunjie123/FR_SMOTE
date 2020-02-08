from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt  # 导入模块
import numpy as np


data = []
for i in range(80):
    data.append(random.normalvariate(0, 1))
    data.append(random.normalvariate(4, 1))
# plt.hist(data, bins=40)  # bins直方图的柱数
# plt.show()

data = np.array(data).reshape(-1, 1)

#返回一个带有分类标签的矩阵
def kmeans_class(data, center_number):
    km = KMeans(n_clusters=center_number, random_state=9)
    y = km.fit_predict(data)
    c_data = np.c_[data, y]
    return c_data

#返回二分类分类的边界
def kmeans_border(data):
    a1 = []
    a2 = []
    border = []
    thread = 0
    for x in data:
        if x[1] == 1:
            a1.append(x[0])
        else:
            a2.append(x[0])
    border.append(max(a1))
    border.append(min(a1))
    border.append(max(a2))
    border.append(min(a2))
    border = np.sort(border, axis=0)
    thread = (border[1] + border[2])/2
    return thread

# def danger_point(thread1, thread2):
