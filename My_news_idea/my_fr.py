import My_FR.my_FruchtermannReingold as My_FR
import numpy as np
import pandas as pd
#对原fr算法进行多维化改造
class multi_d_FR(My_FR.FruchtermannReingold):

    def __init__(self, title, vertices, edges, iterations, temperature,
                 attractive_force, repulsive_force, speed):
        super().__init__(title, vertices, edges, iterations, temperature,
                         attractive_force, repulsive_force, speed)

    def algorithm_step(self):
        # Initialization of forces
        m, n = self.vertices.shape
        for i in range(m):
            f_node = [0 for _ in range(n)]
            self.forces[i] = f_node   #形成一个字典，键是顶点，值是f_node
        # Calculation repulsion forces
        for i in range(m):
            vertex_1 = i
            for j in range(i + 1, m):
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

        #更新位置
        for v in range(m):
            disp = self.forces[v]
            mod_disp = self.norm(disp)
            # print("position_head:", self.positions[v])
            self.positions[v] = self.sum(self.positions[v], self.mult(
                self.div(disp, mod_disp),
                min(mod_disp, self.temperature)))
            # print("position_end:", self.positions[v])
        # Cool
        self.temperature = self.cool()

    def draw(self):
        self.init_vertices()  #初始化顶点的坐标
        for i in range(0, self.iterations):
            self.algorithm_step()
        value = np.array([v for v in self.positions.values()])
        return value
