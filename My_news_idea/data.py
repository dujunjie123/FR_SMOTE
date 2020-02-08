import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import My_FR.my_FruchtermannReingold as my_fr
import My_FR.tool as tool
data = []
for i in range(500):
    data.append(random.normalvariate(0, 1))
    data.append(random.normalvariate(0, 1))
for j in range(1500):
    data.append(random.normalvariate(4, 1))
    data.append(random.normalvariate(4, 2))

data = np.array(data).reshape(2000, 2)
def draw(data):
    plt.figure()
    plt.scatter(data[0:500:, 0], data[0:500, 1])
    plt.scatter(data[500:1500:, 0], data[500:1500:, 1])
    plt.show()

draw(data)

clustering = DBSCAN(eps=0.1, min_samples=3).fit(data)
# print(type(clustering.labels_))
# index = np.argwhere(clustering.labels_ == 0).reshape(1, -1)[0]
# print(index)
# print(data[index].shape)

mean_list = []
for i in clustering.labels_:
    indexs = np.argwhere(clustering.labels_ == i).reshape(1, -1)[0]
    # print(data[indexs].shape)
    x_list = [x[0] for x in data[indexs]]
    y_list = [x[1] for x in data[indexs]]
    means = [np.mean(x_list), np.mean(y_list)]
    mean_list.append(means)

print(mean_list)

def fr(data):
    size_of_data = len(data)
    np.set_printoptions(suppress=True)
    new_dist = tool.cal_dist(np.array(data).reshape(size_of_data, 2))
    E = tool.k_edge(5, new_dist)
    graph_data = (data, E)
    graph = my_fr.FruchtermannReingold('', graph_data[0], graph_data[1], 30, 5, 10, 0.4, 0.02)
    # 新的数据
    value = graph.draw()
    print(value)

# fr(mean_list)

