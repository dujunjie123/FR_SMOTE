#!/usr/bin/python
import math
import numpy as np
import My_FR.tool as tool
import paper.mds2 as mds
import time


class FruchtermannReingold:
    # vertices:不带标签的np数据
    # position：{0:[1,3],1:[0.5,0.4],...}
    def __init__(self, title, vertices, edges, iterations, temperature, 
        attractive_force, repulsive_force, speed):
        self.title = title
        self.vertices = vertices
        self.edges = edges
        self.iterations = iterations
        self.attractive_force = attractive_force #引力常数
        self.repulsive_force = repulsive_force   #斥力常数
        self.positions = {}
        self.forces = {}
        self.temperature = temperature
        self.plot = None
        self.created = False
        self.speed = speed

    def calculate_attraction_force(self, value):
        return value ** 2 / self.attractive_force

    def calculate_repulsion_force(self, value):
        if value != 0:
            v = self.repulsive_force ** 2 / value
        else:
            v = 0
        return v

    def init_vertices(self):
        # Initialization of vertices positions
        to_ret = []
        for i in range(0, len(self.vertices)):
            to_ret.append((i, self.vertices[i]))
        self.positions = dict(to_ret)  #数据结构如下：{0:[1,3],1:[0.5,0.4],...}

    def cool(self):
        return self.temperature * 0.8

    #求向量的各个元素平方和的开方
    def norm(self, x):
        return math.sqrt(sum(i ** 2 for i in x))

    #把两个向量的分量分别相加
    def sum(self, v1, v2):
        return [x + y for (x, y) in zip(v1, v2)]

    def sub(self, v1, v2):   #计算两个向量的差
        return [x - y for (x, y) in zip(v1, v2)]

    #力方向*力的大小
    # def mult(self, v1, scalar):
    #     return [x * scalar for x in v1]
    def mult(self, v1, scalar):
        return [v1[0] * scalar, v1[1] * scalar*2]

    # 向量单位化，也就是得到力的方向
    # 这个地方可以控制力的方向！！！！！！！！！！！！！！！！！！！！！
    def div(self, v1, scalar):
        if scalar != 0:
            v = [x / scalar for x in v1]
        else:
            v = [x*0 for x in v1]
        return v

    def algorithm_step(self):
        # Initialization of forces
        for i in range(0, len(self.vertices)):
            f_node = [0.0, 0.0]
            self.forces[i] = f_node   #形成一个字典，键是顶点，值是f_node
        # Calculation repulsion forces
        for i in range(len(self.vertices)):
            vertex_1 = i
            for j in range(i + 1, len(self.vertices)):
                vertex_2 = j
                # self.positions 数据结构如下：{0:[1,3],...}
                delta = self.sub(self.positions[vertex_1], self.positions[vertex_2])#向量相减
                mod_delta = self.norm(delta) #取向量分量平方和的开方与0.02中最大的一个
                #calculate_repulsion_force：计算斥力大小
                #div:斥力的方向
                #mult:计算斥力
                #sum:计算原来的向量所受的力
                self.forces[vertex_1] = self.sum(self.forces[vertex_1], \
                    self.mult(self.div(delta, mod_delta), self.calculate_repulsion_force(mod_delta))
                )
                #在力的作用下，两个向量的移动方向是相反的
                self.forces[vertex_2] = self.sub(self.forces[vertex_2], \
                    self.mult(self.div(delta, mod_delta), \
                    self.calculate_repulsion_force(mod_delta))
                )
        # Calculation attraction forces
        # edge数据结构：[[v1,v2],...] 表示有边的向量对
        for edge in self.edges:
            # 在k近邻之间计算相应的引力
            delta = self.sub(self.positions[edge[0]], self.positions[edge[1]])
            mod_delta = self.norm(delta)
            self.forces[edge[0]] = self.sub(self.forces[edge[0]], 
                self.mult(self.div(delta, mod_delta), 
                self.calculate_attraction_force(mod_delta))
            )
            self.forces[edge[1]] = self.sum(self.forces[edge[1]], 
                self.mult(self.div(delta, mod_delta), 
                self.calculate_attraction_force(mod_delta))
            )
        # Update positions
        # for vertex in self.vertices:
        #     disp = self.forces[vertex]
        #     mod_disp = max(self.norm(disp), 0.02)
        #     self.positions[vertex] = self.sum(self.positions[vertex], self.mult(
        #             self.div(disp, mod_disp), min(mod_disp, self.temperature))
        #     )

        for v in range(m):
            disp = self.forces[v]
            mod_disp = self.norm(disp)
            self.positions[v] = self.sum(self.positions[v], self.mult(
                self.div(disp, mod_disp), min(mod_disp, self.temperature)))
        # Cool
        self.temperature = self.cool()

    def __draw__(self):

        values = np.array([v for v in self.positions.values()])
        values_len = values.shape[0]
        values = values.reshape(values_len, 2)
        tool.draw(values)
        return values


    def draw(self):
        self.init_vertices()  #初始化顶点的坐标
        self.__draw__()
        for i in range(0, self.iterations):
            self.algorithm_step()
            # self.__draw__()
            # time.sleep(1)
        value = self.__draw__()
        return value

# 返回：标题，顶点，边缘的元组
# "D:/pythonWorkspace/data/iris.csv"
# k 是k近邻  dir是读取数据的地址
# V是数据  E是有边的数据序号对
def read_graph_data(k, dir):
    V = tool.my_data(dir)
    E = tool.my_k_edge(k, dir)
    return (V, E)

# def add_arguments():     #从命令行中读取以   值键对出现的参数，并返回相应的parser.parse_args()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--iterations', type = int, default = 50,
#         help = 'Cantidad de iteraciones (default 100)')
#     parser.add_argument('-t', '--temperature', type = int, default = 200,
#         help = 'Temperatura para el algoritmo. Afecta al movimiento de los\
#         vertices y de la terminacion del algoritmo')
#     parser.add_argument('-af', '--attractive-force', type = float,
#         default = 2000.0, help = 'Constante de las fuerzas de atraccion entre\
#         2 vertices')
#     parser.add_argument('-rf', '--repulsive-force', type = float,
#         default = 30.0, help = 'Constante de las fuerzas de atraccion entre\
#         2 vertices')
#     parser.add_argument('-s', '--speed', type = int, default = 100,
#         help = 'Velocidad de animacion')
#     return parser.parse_args()

def main():
    # args = add_arguments()  #从命令行中读取相应的参数
    graph_data = read_graph_data(0.1, "D:/pythonWorkspace/data/iris.csv")
    graph = FruchtermannReingold('', graph_data[0], graph_data[1], 30, 1, 1, 1, 0.02)
    graph.draw()

def main2():
    np.set_printoptions(suppress=True)
    data = tool.my_data("D:/pythonWorkspace/data/iris.csv")
    dist = tool.cal_dist(np.array(data).reshape(150, 4))
    print(dist[:10:, :10:])
    #降维
    V = np.real(mds.MDS(dist, 2)).tolist()
    new_dist = tool.cal_dist(np.array(V).reshape(150, 2))
    # print(new_dist[:9:, :9:]-dist[:9:, :9:])
    E = tool.k_edge(0.5, new_dist)
    graph_data = (V, E)
    graph = FruchtermannReingold('', graph_data[0], graph_data[1], 30, 5, 5, 4, 0.02)
    #新的数据
    value = graph.draw()
    print(value)


# main2()
# def test():
#     x1 = np.random.normal(0, 1, 50)
#     y1 = np.random.normal(0, 1, 50)
#     x2 = np.random.normal(10, 3, 50)
#     y2 = np.random.normal(7, 3, 50)
#     x = np.append(x1, x2)
#     y = np.append(y1, y2)
