import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn.cluster import DBSCAN
import My_news_idea.my_fr as my_fr
import My_FR.tool as tool
from imblearn.over_sampling import BorderlineSMOTE as B_SMOTE
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix

# 获取数据的中心点，并给中心点添加标签
# data :
def mean_list(data):
    mean_list = []
    #获取密度聚类的模型
    clustering = DBSCAN(eps=0.2, min_samples=3).fit(data[:, 0:2])
    #print("clustering.labels_", clustering.labels_)
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



# 中心点数据
# data 为 dataframe 数据
# eps:点之间的最大聚类距离
# min_samples : 最小簇的样本数
# return 返回各个簇中心点的坐标和标签 返回数据的数据类型(np)[[1,2,1],[0,3,0],.....]
# 只能进行二分类
def get_cent_point(data, eps=0.0001, min_samples = 1):
    np.set_printoptions(suppress=True)
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
    return np.around(mean_list, decimals=5)


# print("cent_point:", cent_point)
# print("k近邻：", tool.get_edge(pd.DataFrame(cent_point),0.9))

# 力导模型
# data: 带标签的np 数据,
# 输出的带标签的dataframe 数据
# terations：迭代次数, temperature=退火的温度,
# attractive_force：引力, repulsive_force：斥力, speed：迭代的次数
# k:k近邻

def fr(data, iterations=30, temperature=5, attractive_force=10, repulsive_force=0.4, speed=0.02, k=0.5):
    np.set_printoptions(suppress=True)
    E = tool.get_edge(pd.DataFrame(data), k)
    graph_data = (data[:, 0:-1], E)
    graph = my_fr.multi_d_FR('', graph_data[0], graph_data[1], iterations, temperature,
                             attractive_force, repulsive_force, speed)
    # 新的数据
    # print("标签列：", data[:, -1])
    # value = graph.draw()

    value = np.c_[graph.draw(), data[:, -1]]
    return value




# 使用平衡过后的数据集训练树模型
# my_tree = tree.DecisionTreeClassifier()
#
# base_tree = my_tree.fit(fr_data_x_smote, fr_data_y_smote)
#
# print("混淆矩阵：", confusion_matrix(Y_test, base_tree.predict(X_test)))




# 获取使用力导模型和smote处理后的中心点数据
# mean_list : 中心点的列表
# def get_worked_data(mean_list):



#训练模型，并预测
def train(x_train, y_train, x_test, y_test):
    my_tree = tree.DecisionTreeClassifier()
    base_tree = my_tree.fit(x_train, y_train)
    baseline = confusion_matrix(y_test, base_tree.predict(x_test))
    print(baseline)


# path:数据的路径
# began：数据的起始行数
# is_cent_data : 判断是否对数据进行中心化处理,默认0 不进行中心化处理
def main(path, began, is_cent_data=0, iterations=30, temperature=5, attractive_force=1, repulsive_force=0.4, speed=0.02, k=0.5):
    # 读取数据
    # 读取df类型的归一化数据
    my_data = tool.unitilize_data(tool.read_KEEL_data(path, began))
    # 准备训练数据和测试数据
    # 使用fr模型和smote模型数理数据，形成新的数据
    # 使用处理完成的数据，进行建模，并预测
    # 是否对数据进行中心化处理
    if is_cent_data != 0:
        cent_point = get_cent_point(my_data)
    else:
        cent_point = np.array(my_data)
    fr_data = pd.DataFrame(fr(cent_point,iterations, temperature, attractive_force, repulsive_force, speed, k))
    fr_data_x = fr_data.iloc[:, 0:-1]
    fr_data_y = fr_data.iloc[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # 随机抽取测试数据的训练数据
    X_train, X_test, Y_train, Y_test = train_test_split(fr_data_x, fr_data_y, test_size=0.3, random_state=42)
    # 根据抽取的训练数据，生成平衡过后的数据集
    my_B_SMOTE = B_SMOTE()
    fr_data_x_smote, fr_data_y_smote = my_B_SMOTE.fit_sample(X_train, Y_train)

    #经过处理的数数据
    print("经过处理的数据:")
    train(fr_data_x_smote, fr_data_y_smote, X_test, Y_test)

    #未经处理的数据

    X_train_org, X_test_org, Y_train_org, Y_test_org = train_test_split(my_data.iloc[:, 0:-1], my_data.iloc[:, -1], test_size=0.3, random_state=42)

    data_x_org, data_y_org = my_B_SMOTE.fit_sample(X_train_org,  Y_train_org)
    print("未处理的数据:")
    train(data_x_org, data_y_org, X_test_org, Y_test_org)

if __name__ == '__main__':
    main("C:\\Users\\Administrator\\Desktop\\keel\\glass1.dat", 14, )