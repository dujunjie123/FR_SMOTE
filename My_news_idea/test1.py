import numpy as np
import pandas as pd
a = [[1, 2, 3], [4, 5, 6],[7, 8, 9]]
a = np.array(a)
a = pd.DataFrame(a)
print(a)
print(a.shape)
print(np.array(a.iloc[1, 0:-1]))
print(np.linalg.norm(a.iloc[1, 0:-1]-a.iloc[0, 0:-1]).__class__)
# print(a)
# print(np.array(a.iloc[b]).tolist())
#
# print(a.columns.tolist())
#计算dataframe 每列的均值
#return 均值的列表
def col_df_mean(df):
    col = df.columns.tolist()
    mean_list = []
    for c in col:
        if np.mean(np.array(df[c])) > 0:
            print("s")

        mean_list.append(np.mean(np.array(df[c])))
    return mean_list

