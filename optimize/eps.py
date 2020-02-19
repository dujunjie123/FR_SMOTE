#自适应的获得数据集的eps和数据集的核心点
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import invgauss
from scipy.stats import norm

data = [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]

data_df = pd.DataFrame(data)
print(data_df)
# 获得数据点的距离矩阵（[[(n,dist),(),...]
#                      [(),(),...]]）
# data 是dataframe带标签数据，并且进行过归一化的处理
def get_dist(data):
    n, m = data.shape
    dist = np.zeros((n, n, 2), np.float16)
    for x in range(n):
        for y in range(0, x):
            temp_dist = np.linalg.norm(data.iloc[x, 0:-1] - data.iloc[y, 0:-1])
            dist[x][y] = [x, temp_dist]
            dist[y][x] = [x, temp_dist]
    # print("dist:", dist)
    return dist

# dist n 维np 数据

def sort_by_dist(dist):
    n = dist.shape[0]
    for i in range(n):
        col = dist[:, i]
        dist[:, i] = col[np.argsort(col[:, -1])]
    return dist.T

# 获得转置sorted_dist 矩阵的指定行
# dist 排序后的距离矩阵
# row_num 行号
def get_row(dist, row_num):
    return dist[row_num, 1:]


# 得到数据的柱形图
# row_list : np数据
# bin : 柱形图的x轴分成对少块
# return : x 横坐标,y：纵坐标(np)

def get_hist(row_list, bin):
    np.set_printoptions(precision=3)
    cnt = plt.hist(row_list, bins=bin)
    for i in range(1, len(cnt[0])):
        cnt[0][0] = cnt[0][0] + cnt[0][i]
    y = np.array(cnt[0][0])
    x = []
    for i in range(len(cnt[1]) - 1):
        x.append((cnt[1][i] + cnt[1][i + 1]) / 2)
    x = np.array(x)
    return x, y


# 根据距离矩阵的行数据，生成相应的分布，并且求出峰值
# row_list 矩阵的某一行数据
def get_peak(row_list, bin):
    x, y = get_hist(row_list, bin)
    print(invgauss.fit(y))




# get_peak(pd.DataFrame(get_row(sort_by_dist(get_dist(data_df))[1], 1)))

list0 = [1,3,1,2,4,2,3,4,2,3,4,2,3,4,1,2,8,4,1,2,3,1,2,3,1,2,3,1,2,4,1,2,4,1,2,3,1,2,3,1,2,4,1,2,4]
list1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
list = pd.DataFrame(list1)

get_peak(list, 5)

