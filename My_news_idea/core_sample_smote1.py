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

data = []
for i in range(200):
    data.append(random.normalvariate(1, 0.5))
    data.append(random.normalvariate(1, 1))
    data.append(0)
for j in range(400):
    data.append(random.normalvariate(5, 1))
    data.append(random.normalvariate(4, 2))
    data.append(1)
for j in range(500):
    data.append(random.normalvariate(1, 1))
    data.append(random.normalvariate(8, 2))
    data.append(1)
for j in range(500):
    data.append(random.normalvariate(-1, 0.5))
    data.append(random.normalvariate(5, 2))
    data.append(1)
for j in range(400):
    data.append(random.normalvariate(3, 0.5))
    data.append(random.normalvariate(6, 2))
    data.append(1)
data = np.array(data).reshape(2000, 3)


test_data = []
for i in range(50):
    test_data.append(random.normalvariate(0, 1))
    test_data.append(random.normalvariate(0, 1))
    test_data.append(0)
for j in range(150):
    test_data.append(random.normalvariate(4, 1))
    test_data.append(random.normalvariate(4, 2))
    test_data.append(1)
test_data = np.array(test_data).reshape(200, 3).tolist()


def draw(data):
    plt.figure()
    plt.scatter(data[0:200:, 0], data[0:200, 1])
    plt.scatter(data[200:2000:, 0], data[200:2000:, 1])
    plt.show()

#原始数据
draw(data)

def mean_list(data):
    print("mean_list")
    mean_list = []
    clustering = DBSCAN(eps=0.2, min_samples=3).fit(data[:, 0:2])
    print("clustering.labels_",clustering.labels_)
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

#这个数据是对原始数据进行聚类后，得到聚类中心
mean_data = mean_list(data)
# print("mean_data len:", len(mean_data))

#这是原始数据
# mean_data = data
#使用力导模型处理数据
fr_data = fr(np.array(mean_data).reshape(len(mean_data), 3)[:, 0:2].tolist())
fr_data_list = fr_data.tolist()
for i in range(len(fr_data_list)):  #把新生成的力导模型和标签叠加在一起，形成移动过后的带有标签的中心点
    fr_data_list[i].append(mean_data[i][2])
len_fr = len(fr_data_list)
fr_data_list = np.array(fr_data_list).reshape(len_fr, 3)

#经过力导模型处理后的数据
# plt.figure()
# plt.scatter(fr_data_list[:, 0], fr_data_list[:, 1])
# plt.show()

#转换测试数据
new_test_data = newData.convert_text_data(test_data, mean_data, fr_data_list.tolist())
test_data = np.array(new_test_data)
print(test_data)
train1 = DataFrame(mean_data)
train_fr = DataFrame(fr_data_list)
test = DataFrame(test_data)

plt.figure()
plt.scatter(train1.iloc[:,0], train1.iloc[:,1])
plt.show()

X_train1 = train1.iloc[:, :2]
y_train1 = train1.iloc[:, 2]

x_train_fr = train_fr.iloc[:, :2]
y_train_fr = train_fr.iloc[:, 2]

X_test = test.iloc[:, :2]
y_test = test.iloc[:, 2]

print(len(X_train1))
model_smote = SMOTE() # 建立SMOTE模型对象
x_smote_resampled, y_smote_resampled = model_smote.fit_sample(X_train1, y_train1) # 输入数据并作过抽样处理

#使用fr模型生成的数据进行训练
model_smote_fr = SMOTE() # 建立SMOTE模型对象
x_smote_resampled_fr, y_smote_resampled_fr = model_smote_fr.fit_sample(x_train_fr, y_train_fr)

#画出过采样的图形
plt.figure()
plt.scatter(x_smote_resampled_fr.iloc[:, 0], x_smote_resampled_fr.iloc[:, 1])
plt.show()

tree1 = tree.DecisionTreeClassifier()
base_tree1 = tree1.fit(x_smote_resampled, y_smote_resampled)
# base_tree2 = tree.fit(X_train1, y_train1)
baseline1 = confusion_matrix(y_test, base_tree1.predict(X_test))
print(baseline1)
# baseline2 = confusion_matrix(y_test, base_tree2.predict(X_test))
# print(baseline2)


#训练fr数据
tree_fr = tree.DecisionTreeClassifier()
base_tree1 = tree_fr.fit(x_smote_resampled_fr, y_smote_resampled_fr)
# base_tree2 = tree.fit(X_train1, y_train1)
baseline2 = confusion_matrix(y_test, base_tree1.predict(X_test))
print(baseline2)