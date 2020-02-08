import numpy as np

# 这个是MDS算法的实现
# 其中D为n*n  np矩阵，d为降维后数据的维数
# 输出的结果是n*d的矩阵，其中n为样本的个数，d为样本的维数

def MDS(D, d):
    DSquare = D ** 2  #每个元素的平方
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis=0)
    rowMean = np.mean(DSquare, axis=1)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    eigVal, eigVec = np.linalg.eig(B)  # 求特征值及特征向量
    # 对特征值进行倒序排序，得到排序索引
    eigValSorted_indices = np.argsort(eigVal)[::-1]
    # 提取d个最大特征值对应的特征向量
    topd_eigVec = eigVec[:, eigValSorted_indices[:d:]]
    # 根据特征值和特征向量求取降维后的数据
    X = np.dot(topd_eigVec, np.sqrt(np.diag(eigVal[eigValSorted_indices[:d:]])))
    return X




