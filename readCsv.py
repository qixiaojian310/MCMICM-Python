# 目标，读取csv文件到dataframe中，使用pandas
from networkx.algorithms.link_analysis.pagerank_alg import google_matrix, pagerank
from numpy.core.fromnumeric import shape
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from cyberbrain import trace


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
    # S = M
    # print(shape(S))
    # for j in range(N):
    #     sum_of_col = sum(S[:, j])
    #     # print('第%d列和:' %j, sum_of_col)S
    #     for i in range(N):
    #         if sum_of_col != 0:
    #             k = S[i, j]/sum_of_col
    #             print('第%d个数:' %j, S[i][j])
    #         else:
    #             S[i, j] = 0
    return M



# 'D:\\MCM\\pythonCode\\data\\influence_dataE.csv'
path = 'E:\\pythonStudy\\MCMICM\\data\\influence_data.csv'
data = pd.read_csv(path)
dataSelect = data[['influencer_id','follower_id']]
dataSelectArray= np.array(dataSelect)

list1 = dataSelect["influencer_id"].values.tolist()

series = dict(zip(*np.unique(list1, return_counts=True)))

# series.items(),可以将字典转换为可迭代元素
seriesSorted = sorted(series.items(), key=lambda x:x[1])
seriesSortedList = [i[0] for i in seriesSorted]
# series = dataSelect[dataSelect.duplicated(['influencer_id'])].count()


# 目标，建立有向网络的数据结构
G = nx.DiGraph()
G.add_edges_from(dataSelectArray)
tableHeader = list(G.nodes())
# 读取节点总数
N = len(tableHeader)


# google matrix
# 邻接矩阵
adjMatrix = nx.to_numpy_matrix(G, nodelist=tableHeader)
# print(adjMatrix)
# 生成谷歌矩阵（自己的
googleMatrix = genGoogleMatrix(M=adjMatrix, N=N)
# googleMatrix = adjMatrix
print('生成谷歌矩阵',type(googleMatrix))
# googleMatrix = nx.google_matrix(G, alpha=1).T
print(googleMatrix)

# 去除掉错误点
prepGoogleMatrix = np.where(googleMatrix > 0.000179, googleMatrix, 0)
print(prepGoogleMatrix)

# 绘制googlematrix的csv
matrixDataFrame = pd.DataFrame(googleMatrix, columns=tableHeader, index=tableHeader)
matrixDataFrame.to_csv('./googleMatrix3.csv',  columns=tableHeader, index=tableHeader)

# 自己的PageRank实现
# 初始化迭代过程
# M = prepGoogleMatrix
# eigenvalues, eigenvectors = np.linalg.eig(M.T)
# ind = np.argmax(eigenvalues)
# largest = np.array(eigenvectors[:, ind]).flatten().real
# norm = float(largest.sum())
# resultDict = dict(zip(G, map(float, largest / norm)))

pN = np.ones(N)
pN2 = np.zeros(N)
# e = 100000 # 误差初始化\
k = 0   # 记录迭代次数
print('loop for pagerank...')
while k < 2:   # 开始迭代
    pN2 = np.dot(googleMatrix, pN)   # 迭代公式
    e = pN2-pN
    e = max(map(abs, e))    # 计算误差
    pN = pN2
    k += 1
    print('iteration %s:' %str(k), pN2)
print('final result:', pN)

pNList = list(pN)
pNdict = dict(zip(tableHeader,pNList))
print(pNdict)
# @trace
# def test():
#     resultDataFrame = pd.DataFrame(pN, index=tableHeader)
#     resultDataFrame.to_csv('./resultMatrix.csv', index=tableHeader)

# test()

# # pagerank networkx实现
# pagerank_list = nx.pagerank(G, alpha = 1)
# pagerank_list_sort = sorted(pagerank_list.items(), key=lambda x: x[1])
# testDataFrame = pd.DataFrame(pagerank_list_sort)
# testDataFrame.to_csv('./resultMatrix.csv')
# print("pagerank 值是：", pagerank_list_sort)


def show_graph(graph, layout='spring_layout'):
    # 使用 Spring Layout 布局，类似中心放射状
    if layout == 'circular_layout':
        positions=nx.circular_layout(graph)
    else:
        positions=nx.spring_layout(graph)
    # 设置网络图中的节点大小，大小与 pagerank 值相关，因为 pagerank 值很小所以需要 *20000
    nodesize = [x['pagerank']*2 for v,x in graph.nodes(data=True)]
    # 设置网络图中的边长度
    # edgesize = [np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]
    # 绘制节点
    nx.draw_networkx_nodes(graph, positions, node_size=nodesize, alpha=0.4)
    # 绘制边
    nx.draw_networkx_edges(graph, positions, alpha=0.2)
    # 绘制节点的 label
    nx.draw_networkx_labels(graph, positions, font_size=10)
    # 输出希拉里邮件中的所有人物关系图
    plt.show()

# # 设置阈值，只画出来核心节点G
pagerank_threshold = 0.005
nx.set_node_attributes(G, name = 'pagerank', values=pNdict)
sG = G.copy()
for n, p_rank in G.nodes(data=True):
    if p_rank['pagerank'] < pagerank_threshold:
        sG.remove_node(n)

show_graph(sG,'spring_layout')
