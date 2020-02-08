import numpy as np

#判断实验样本属于哪个核心点的成员
#找到与实验样本最近的核心点，然后以对应的fr_core_data点为圆心，以dist为半径，随机移动
def convert_text_data(text_data, core_data, fr_core_data):
    new_data_list = []
    for every in text_data:
        dest = []
        for point in core_data:
            dest.append(np.sqrt(np.sum((np.array(every[0:2]) - np.array(point[0:2])) ** 2)))
        #最小距离
        min_dist = min(dest)
        #最小距离的坐标
        min_index = dest.index(min_dist)
        #找到fr变换之后的最近的中心点
        nei_fr_core_data = fr_core_data[min_index][0:2]
        nei_fr_core_data.append(every[-1])
        new_data_list.append(nei_fr_core_data)
    return new_data_list

# def random_gen_data(min_dist, nei_fr_core_data):
#     new_point = []
#     r = np.random.normal(size=(1, 2))
#


