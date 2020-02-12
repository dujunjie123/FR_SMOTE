import numpy as np
import My_FR.tool as tool
import pandas as pd

data = tool.read_KEEL_data("C:\\Users\\Administrator\\Desktop\\keel\\glass1.dat",14)
print(data.shape)
print(data.head(100).iloc[:, 0:-1])
print(pd.get_dummies(data).head(100).iloc[:, 0:-1])