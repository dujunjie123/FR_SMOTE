import numpy as np
import pandas as pd
a = [[1, 2, 3], [4, 5, 6],[7, 8, 9],[7, 8, 9]]
a = np.array(a)
p = pd.DataFrame(a)
print(a.__class__)
print(len(a))

# print(a)
# print(np.array(a.iloc[b]).tolist())
#
# print(a.columns.tolist())
#计算dataframe 每列的均值
#return 均值的列表

class A_c():

    def __init__(self):
       self.my_print()

    def my_print(self):
        print("test")


c = A_c()
