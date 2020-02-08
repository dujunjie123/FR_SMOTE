import csv
import numpy as np
# 读取数据
my_data = []
with open("D:/pythonWorkspace/data/iris.csv") as iris:
    csv_reader = csv.reader(iris)
    for row in csv_reader:
        my_data.append(row)
my_data = np.array(my_data).reshape(151, 6)


# 获取训练数据，不带分类信息

my_float_data = []
for x in my_data[1::, 0:-1:]:
    temp = [float(m) for m in x]
    my_float_data.append(temp)
my_float_data = np.array(my_float_data).reshape(150, 5)
# print(np.size(my_float_data, 0))


# 0# 计算距离矩阵
dist = np.zeros((150, 150), np.float16)
for x in range(150):
    for y in range(0, x+1):
        temp_dist = np.linalg.norm(my_float_data[x][1::]-my_float_data[y][1::])
        dist[x][y] = temp_dist
        dist[y][x] = temp_dist

        # print(my_float_data[x][1::]-my_float_data[y][1::])
print(dist)

# 计算k近邻  以距离 1 举例
k_list = []
for x in range(150):
    k_member = []
    for y in range(150):
        if dist[x][y] < 1:
            k_member.append(y)
    kv = {x: k_member}
    k_list.append(kv)
print(k_list)






