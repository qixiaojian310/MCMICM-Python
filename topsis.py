import networkx as nx
from queue import Queue
from networkx.algorithms.efficiency_measures import efficiency
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn import preprocessing
import matplotlib.pyplot as plt


def readCsv():
    path = 'E:\\pythonStudy\\MCMICM\\data\\influence_data.csv'
    return pd.read_csv(path)

data = readCsv()
dataSelect = data[['influencer_id', 'follower_id']]
# 影响者的id和流派
nodeInSelect = data[['influencer_id', 'influencer_main_genre']]
nodeInSelectArray = np.array(nodeInSelect).tolist()
# 粉丝的id和流派
nodeFolSelect = data[['follower_id', 'follower_main_genre']]
nodeFolSelectArray = np.array(nodeFolSelect).tolist()
# 所有音乐家的id和流派
preArtistSelectArray = nodeFolSelectArray
# 得到所有音乐家的流派信息
preArtistSelectArray.extend(nodeInSelectArray)

def genGoogleMatrix(M, N):

    M = np.array(M,dtype='float')
    sum = M.sum(axis=0)
    for i in range(N):
        print(sum[i])
        if sum[i] != 0:
            print(M[:, i])
            M[:, i] /= sum[i]
        else:
            M[:, i] = 0
    return M


def readCsv():
    path = 'E:\\pythonStudy\\MCMICM\\data\\influence_data.csv'
    return pd.read_csv(path)


data = readCsv()
dataSelect = data[['influencer_id', 'follower_id']]
dataSelectArray = np.array(dataSelect)

# 记录节点和他们的流派信息
nodeSelect = data[['influencer_id']]
# 'D:\\MCM\\pythonCode\\data\\influence_data.csv'
# 将影响者的id转换为列表
list1 = dataSelect["influencer_id"].values.tolist()
# 将列表和他们出现的次数结合生成字典
series = dict(zip(*np.unique(list1, return_counts=True)))
# series.items(),可以将字典转换为可迭代元素，并且根据count排序
seriesSorted = sorted(series.items(), key=lambda x: x[1])
seriesSortedList = [i[0] for i in seriesSorted]

# 目标，建立有向网络的数据结构
G = nx.DiGraph()
G.add_edges_from(dataSelectArray)
tableHeader = list(G.nodes())
# 读取节点总数
N = len(tableHeader)

# 邻接矩阵
adjMatrix = nx.to_numpy_matrix(G, nodelist=tableHeader)
# 生成谷歌矩阵（自己的
googleMatrix = genGoogleMatrix(M=adjMatrix, N=N)
print('生成谷歌矩阵', type(googleMatrix))
print(googleMatrix)

# 绘制googlematrix的csv
matrixDataFrame = pd.DataFrame(googleMatrix, columns=tableHeader, index=tableHeader)
matrixDataFrame.to_csv('./googleMatrix3.csv', columns=tableHeader, index=tableHeader)

# 迭代三次得到pagerank的值
pN = np.ones(N)
pN2 = np.zeros(N)
k = 0   # 记录迭代次数
print('loop for pagerank...')
while k < 2:   # 开始迭代
    pN2 = np.dot(googleMatrix, pN)   # 迭代公式
    pN = pN2
    k += 1
    print('iteration%s:' %str(k), pN2)

# PN为最终的pagerank的值
print('final result:', pN)
resultDataFrame = pd.DataFrame(pN, index=tableHeader)
resultDataFrame.to_csv('./resultMatrix.csv', index=tableHeader)

# 设定排除范围
pagerank_threshold = 10
resultTupleDict = dict(zip(tableHeader, pN))
nodeRemove = []
# 遍历字典的值，如果小于要求就记下对应的点
for i in resultTupleDict.items():
    if i[1] < pagerank_threshold:
        nodeRemove.append(i[0])

artistSelectArray = []
for i in preArtistSelectArray:
    if i not in artistSelectArray:
        artistSelectArray.append(i)
        print("加入项", i)
# 将列表转换为字典
artistSelectDict = dict(artistSelectArray)
print(artistSelectDict)
dataSelectArray = np.array(dataSelect)


# 第二部分
print("计算Betweenness系数")
CB = dict.fromkeys(G, 0.0)
for s in G.nodes():
    # 这里的key是终点，而不是edge的起点，pred反应每个指向点w的点，意思就是从节点s到节点v的最短路径中v的前驱节点集。
    Pred = {w: [] for w in G.nodes()}
    # distance，只有距离为1的才是直接节点
    dist = dict.fromkeys(G, None)
    # σst，也就是从s到其他点的最短路径数量
    sigma = dict.fromkeys(G, 0.0)
    dist[s] = 0
    sigma[s] = 1
    Q = Queue()
    Q.put(s)
    S = []
    while not Q.empty():
        # v是起点，v指向w，v是影响者，w是粉丝
        v = Q.get()
        S.append(v)
        for w in G.neighbors(v):
            if artistSelectDict[w] != artistSelectDict[v]:
                if dist[w] is None:
                    dist[w] = dist[v] + 1
                    Q.put(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    Pred[w].append(v)

    # print(Pred)
    # delta记录了δst(v)=σst(v)/σst，也就是穿过v的st最短路径于所有最短路径的比
    delta = dict.fromkeys(G, 0.0)
    # 切片从最后数到最前面
    for w in S[::-1]:
        for v in Pred[w]:
            delta[v] += sigma[v]/sigma[w]*(1+delta[w])
        if w != s:
            CB[w] += delta[w]

# sklearn归一化
data = []
indexCB = []
for v in CB:
    print("当前节点为%d" %v, "CB为%f" %CB[v])
    indexCB.append(v)
    data.append(CB[v])

data = np.array(data)
CBCSV = pd.DataFrame(data, index=indexCB)
CBCSV.to_csv('./CB1.csv', index=indexCB)

# 第一步sklearn归一化
data = preprocessing.minmax_scale(data)
pN = preprocessing.minmax_scale(pN)
# 创建对应的dataFrame
pNFrame = pd.DataFrame(pN, index=tableHeader, columns=['PR'])
datFrame = pd.DataFrame(data, index=indexCB, columns=['CB'])
joinFrame = pNFrame.join(datFrame, how='outer')
joinFrame.to_csv('./combine.csv')
print(joinFrame)
# 第二步，将数组合并为一个矩阵
comMatrix = np.array(joinFrame)
# # 标准化矩阵
# normalEff = np.power(np.sum(pow(comMatrix, 2), axis=1), 0.5)
# print(normalEff)
# for i in range(0, normalEff.size):
#     for j in range(0, comMatrix[i].size):
#         if normalEff[i] != 0:
#             comMatrix[i, j] = comMatrix[i, j]/normalEff[i]
#         else:
#             comMatrix[i, j] = 0
# comMatrixDF = pd.DataFrame(comMatrix, index=joinFrame.index.values)
# comMatrixDF.to_csv('./pythonCode/combine.csv')

# 获取最大值最小值
# 获取每一列的最大值
list_max = np.array([np.max(comMatrix[:, 0]), np.max(comMatrix[:, 1])])
# 获取每一列的最小值
list_min = np.array([np.min(comMatrix[:, 0]), np.min(comMatrix[:, 1])])
# 存放第i个评价对象与最大值的距离
max_list = []
min_list = []       #存放第i个评价对象与最小值的距离
answer_list=[]      #存放评价对象的未归一化得分
# 遍历每一列数据
for k in range(0, np.size(comMatrix, axis=0)):
    # print("k为", k)
    max_sum = 0
    min_sum = 0
    # 有两个指标，计算正距离和负距离
    for q in range(0, 2):
        print(comMatrix[k, q],"      ",list_max[q])
        max_sum += np.power((comMatrix[k, q]-list_max[q]), 2)
        min_sum += np.power((comMatrix[k, q]-list_min[q]), 2)
    max_list.append(pow(max_sum, 0.5))
    min_list.append(pow(min_sum, 0.5))
    # 套用计算得分的公式 Si = (Di-) / ((Di+) +(Di-))
    if min_list[k] + max_list[k] != 0:
        efficient = min_list[k]/(min_list[k] + max_list[k])
    else:
        efficient = 0
    answer_list.append(efficient)
    max_sum = 0
    min_sum = 0
# 得分归一化
answer = np.array(answer_list)
print(shape(answer))
finalAnswer = answer/np.sum(answer)
print("最终得分情况", finalAnswer)
finalAnswerDF = pd.DataFrame(finalAnswer, index=joinFrame.index)
print(finalAnswerDF)
finalAnswerDF.to_csv('./finalAnswer.csv')
