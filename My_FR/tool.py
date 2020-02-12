import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
# from sklearn.metrics import precision_recall_curve
# #决策树的基本操作
# from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# 读取CSV文件
# "D:/pythonWorkspace/data/iris.csv"
def read_csv(dir):
    my_data = []
    with open(dir) as iris:
        csv_reader = csv.reader(iris)
        for row in csv_reader:
            my_data.append(row)
    return my_data

#返回不带标签的数据
def data_no_title(data):
    my_float_data = []
    for x in data[1::, 1:5:]:
        temp = [float(m) for m in x]
        my_float_data.append(temp)
    # print("data no title")
    # print(my_float_data)
    return my_float_data

#单位化向量
# vertor是一个
def unitized_vector(vertor):
    for i in range(len(vertor)):
        square = math.sqrt(sum(x**2 for x in vertor[i]))
        vertor[i] = [x/square for x in vertor[i]]
    return vertor

#计算距离矩阵,计算的矩阵是下三角矩阵，因为距离矩阵为对称阵
#data 是dataframe带标签数据
def cal_dist(data):
    n, m = data.shape
    dist = np.zeros((n, n), np.float16)
    for x in range(n):
        for y in range(0, x):
            temp_dist = np.linalg.norm(data.iloc[x, 0:-1] - data.iloc[y, 0:-1])
            dist[x][y] = temp_dist
            dist[y][x] = temp_dist
    print("dist:",dist)
    return dist

#计算k近邻
#k为距离，dist为(np)距离矩阵
#返回：[[NUM_OF_v1,NUM_OF_v2],...]
def k_edge(k, dist):
    E = []
    n, m = dist.shape
    for x in range(n):
        for y in range(0, x):
            if dist[x][y] < k :
                E.append([x, y])
    return E

# 计算距离矩阵,计算的矩阵是下三角矩阵，因为距离矩阵为对称阵
# 并且计算k近邻
# data: df数据类型
# k:k近邻
# return：距离小于k的点对的列表， [[1,4],[3,7],...]
def get_edge(data,k):
    return k_edge(k, cal_dist(data))

# 返回的是相应单位化后的数据
def my_data(dir):
    data = np.array(read_csv(dir)).reshape(151, 6)
    # data = unitized_vector(data_no_title(data))
    data = data_no_title(data)
    return data

# 读取keel中的数据
# path:数据的路径  # "C:\\Users\\Administrator\\Desktop\\keel\\glass1.dat"
# begain:数据开始的行数
def read_KEEL_data(path, begin=0):
    data = pd.read_table(path, sep=",", skiprows=begin, header=None)
    return data

# dataframe数据的单位化
def unitilize_data(dataFrame):
    newDataFrame = pd.DataFrame(index=dataFrame.index)
    columns = dataFrame.columns.tolist()[0:-1]
    tag = dataFrame.columns.tolist()[-1]
    for c in columns:
        d = dataFrame[c]
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    newDataFrame[tag] = dataFrame[tag]
    return newDataFrame

#计算dataframe 每列的均值
#return 均值的列表
def col_df_mean(df):
    col = df.columns.tolist()
    mean_list = []
    for c in col:
        # print(np.mean(np.array(df[c])).__class__)
        mean_list.append(np.mean(np.array(df[c])))
    return mean_list

# 计算df数据的距离矩阵
def cal_df_dist(df):
    n, m = df.shape
    dist = np.zeros((n, n), np.float16)
    for x in range(n):
        for y in range(0, x):
            temp_dist = np.linalg.norm(df.loc[x].values[0:-1] - df.loc[y].values[0:-1])
            dist[x][y] = temp_dist
            dist[y][x] = temp_dist
    return dist


# print(unitilize_data(read_KEEL_data("C:\\Users\\Administrator\\Desktop\\keel\\glass1.dat", 14)))


def my_k_edge(k, dir):
    data = my_data(dir)
    dist = cal_dist(np.array(data).reshape(150, 4))
    E = k_edge(k, dist)
    return E


#绘制二维向量的散点图
# data 是np
def draw(data):
    plt.figure()
    # plt.scatter(data[::, 0], data[::, 1])
    plt.scatter(data[0:200:, 0], data[0:200, 1])
    plt.scatter(data[200:2000:, 0], data[200:2000:, 1])
    # plt.scatter(data[101:150:, 0], data[101:150, 1])
    plt.show()

# draw(np.array(my_data("D:/pythonWorkspace/data/iris.csv")).reshape(150, 2))

#判断两个向量和各个特征边界之间的关系
def position_with_border(v1, v2, border0, border1):
    # 1/abs(v1[0]-border0)+1/abs(v1[1]-border1)
    value00 = (v1[0]-border0)*(v2[0]-border0)
    value11 = (v1[1] - border1)*(v2[1] - border1)
    status = 1/abs(value00) + 1/abs(value11)
    # if value00 < 0 and value11 < 0:
    #     status = 1/abs(value00) + 1/abs(value11)
    # elif value00*value11 < 0 :
    #     status =
    return status

#计算dataframe每一列的均值



# a = my_k_edge(0.3, "D:/pythonWorkspace/data/iris.csv")
# print(a)

# X = np.array(read_csv("D:/pythonWorkspace/data/iris.csv")).reshape(151, 6)[1::, 1:3:]
# Y = np.array(read_csv("D:/pythonWorkspace/data/iris.csv")).reshape(151, 6)[1::, 5:6:]
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
#
# # 核心代码：使用信息熵作为划分标准，对决策树进行训练
# clf = sklearn.tree.DecisionTreeClassifier(criterion='entropy')
# clf.fit(X_train, Y_train)

# # 预测结果
# answer = clf.predict(X_train)
# answer_proba = clf.predict_proba(X_train)  # 计算属于每个类的概率
# print(answer)
# print(answer_proba)
#
# # sklearn中的classification_report函数用于显示主要分类指标的文本报告
# # 具体见https://blog.csdn.net/akadiao/article/details/78788864
# answer = clf.predict(X)
# print(classification_report(Y, answer))

