import networkx as nx
from queue import Queue
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
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

artistSelectArray = []
for i in preArtistSelectArray:
    if i not in artistSelectArray:
        artistSelectArray.append(i)
        print("加入项", i)
# 将列表转换为字典
artistSelectDict = dict(artistSelectArray)
print(artistSelectDict)
dataSelectArray = np.array(dataSelect)



# 目标，建立有向网络的数据结构
G = nx.DiGraph()
G.add_edges_from(dataSelectArray)
#algorithm from networkx 

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

indexCB = []
for v in CB:
    print("当前节点为%d" %v, "CB为%f" %CB[v])
    indexCB.append(v)
CBCSV = pd.DataFrame(CB.values(), index=indexCB)
CBCSV.to_csv('./CB1.csv', index=indexCB)
