import numpy as np
import My_FR.tool as tool

data = tool.read_KEEL_data("C:\\Users\\Administrator\\Desktop\\keel\\glass1.dat",14)
print(data.shape)
print(data.loc[0].values[0:-1])