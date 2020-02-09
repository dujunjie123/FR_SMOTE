import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification

import My_news_idea.convert_test_data as newData

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from pandas.core.frame import DataFrame


import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import My_FR.my_FruchtermannReingold as my_fr
import My_FR.tool as tool


# 获取数据的中心点，并给中心点添加标签
def mean_list(data):
    mean_list = []
    #获取密度聚类的模型
    clustering = DBSCAN(eps=0.2, min_samples=3).fit(data[:, 0:2])
    print("clustering.labels_", clustering.labels_)
    max_cluster = max(clustering.labels_)
    for i in range(max_cluster):
        indexs = np.argwhere(clustering.labels_ == i).reshape(1, -1)[0]
        x_list = [x[0] for x in data[indexs]]
        y_list = [x[1] for x in data[indexs]]
        label = np.mean([x[2] for x in data[indexs]])#这里获取一个聚类中心周围点标签的情况
        mean_label = 0
        if label >= 0.5:  #如果1的标签多于1/2，则中心点就为1
            mean_label = 1
        means = [np.mean(x_list), np.mean(y_list), mean_label]
        mean_list.append(means)
    return mean_list

#力导模型
#输出的不带标签的np数据
def fr(data):
    size_of_data = len(data)
    np.set_printoptions(suppress=True)
    new_dist = tool.cal_dist(np.array(data).reshape(size_of_data, 2))
    E = tool.k_edge(1, new_dist)
    graph_data = (data, E)
    graph = my_fr.FruchtermannReingold('', graph_data[0], graph_data[1], 30, 5, 10, 0.4, 0.02)
    # 新的数据
    value = graph.draw()
    return value

# 中心点数据
# data 为 dataframe 数据
# eps:点之间的最大聚类距离
# min_samples : 最小簇的样本数
# return 返回各个簇中心点的坐标和标签 返回数据的数据类型(np)[[1,2,1],[0,3,0],.....]
# 只能进行二分类
def get_cent_point(data, eps=0.2, min_samples = 3):
    data = pd.get_dummies(data).iloc[:, 0:-1]
    data_point = data.iloc[:, 0:-1]
    data_label = data.iloc[:,-1]
    mean_list = []
    # 获取密度聚类的模型
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_point)
    # print("clustering.labels_", clustering.labels_)
    max_cluster = max(clustering.labels_)
    for i in range(max_cluster):
        indexs = np.argwhere(clustering.labels_ == i).reshape(1, -1)[0]
        x = data_point.iloc[indexs]
        label = np.mean(np.array(data_label[indexs]))  # 这里获取一个聚类中心周围点标签的情况
        cent_label = 0
        # 如果1的标签多于1/2，则中心点就为1
        if label >= 0.5:
            cent_label = 1
        #计算dataframe每一列的均值
        cent_point = tool.col_df_mean(x)
        cent_point.append(cent_label)
        mean_list.append(cent_point)
    return np.array(mean_list)


print(get_cent_point(tool.read_KEEL_data("C:\\Users\\Administrator\\Desktop\\keel\\glass1.dat", 14)))

# 获取使用力导模型和smote处理后的中心点数据
# mean_list : 中心点的列表
# def get_worked_data(mean_list):


#训练模型，并预测
def train(x_train, y_train, x_test, y_test):
    my_tree = tree.DecisionTreeClassifier()
    base_tree = my_tree.fit(x_train, y_train)
    baseline = confusion_matrix(y_test, base_tree.predict(x_test))
    print(baseline)



#path:数据的路径
#began：数据的起始行数
def main(path, began):
    # 读取数据
    # 读取df类型的归一化数据
    my_data = tool.unitilize_data(tool.read_KEEL_data(path, began))
    # 准备训练数据和测试数据


    # 使用fr模型和smote模型数理数据，形成新的数据
    # 使用处理完成的数据，进行建模，并预测

