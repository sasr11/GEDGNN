import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import random
import networkx as nx
import matplotlib.pyplot as plt
import time

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.shape)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.shape
    _, ind = y.max(dim=-1)
    print("ind", ind)
    y_hard = torch.zeros(shape).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    print("y", y)
    print("y_hard", y_hard)
    return y_hard

def gumbel_softmax_test(logits, n, temperature=1):
    y = gumbel_softmax_sample(logits, temperature)
    print(y)

    _, ind = torch.topk(y, n)  # 选出top-n
    y_hard = torch.zeros_like(y)  # 初始化一个与y同形状的全0张量
    y_hard.scatter_(0, ind, 1)  # 根据索引填充

    # 设置关于y_hard和y的梯度
    y_hard = (y_hard - y).detach() + y
    return y_hard

def transform_matrix(matrix):
    n, m = matrix.shape
    
    # 进行n次操作
    for _ in range(n):
        # 找到最大元素的索引
        max_index = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
        max_row, max_col = max_index
        
        # 设置最大元素所在行列的值为0，最大元素自身设置为1
        matrix[max_row, :] = 0
        matrix[:, max_col] = 0
        matrix[max_row, max_col] = 1
    
    return matrix

def apply_gumbel_softmax():
    logits1 = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]], dtype=torch.float32)
    logits2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32)
    logits3 = torch.tensor([0.11, 0.22, 0.33, 0.44, 0.55], dtype=torch.float32)

    print(torch.cat((logits2.repeat(3, 1), logits3.repeat(3, 1)), dim=1))

    print(logits1.repeat(3, 1))
    print(logits1.size())
    print(logits1.squeeze())
    print(logits1.squeeze(-1).size())
    for i in range(5):
        print(gumbel_softmax_test(logits1, 5))
        print("---------------------")

    for i in range(5):
        print(gumbel_softmax(logits1))
        print("---------------------")

# --------------------------------------------------------------------------------------

def mask_operation():
    # 示例：创建一个n*m的张量x
    n, m = 5, 4  # 假设张量x的形状为5x4
    x = torch.arange(n*m).view(n, m)  # 创建一个示例张量
    print("原始张量x:\n", x)
    rows, cols = x.shape[:2]
    print(rows, cols)
    print(x.numel())
    print(x.view(1, x.numel()))
    # 索引张量，形状为n*1，这里用一个示例，表示要归零第2和第4行（索引从0开始）
    index = torch.tensor([0, 1, 0, 1])
    print(index)
    # 将索引张量转换为布尔张量
    mask = index == 1
    mask = mask.expand_as(x)
    print(mask)
    # 使用布尔索引归零对应的行
    print("\n修改后的张量x:\n", x[mask])
    x[mask.expand_as(x)] = 0
    print("\n修改后的张量x:\n", x)

#---------------------------------------------------------------------------------------
def BCELoss():
    p = torch.tensor([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, -1]], dtype=torch.float32)
    t = torch.tensor([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss()
    print(loss(p, t))


# -------------------------------------------------------------------------------------------
def get_date():
    # 获取当前日期和时间
    now = '_'+datetime.now().strftime('%y-%m%d-%H%M')+'.txt'
    print(now)

# -------------------------------------------------------------------------------------------

def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int =20):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()

def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 1.0, n_iter: int = 20, noise: bool = True):
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise)/tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat

def apply_gs():
    p = torch.tensor([[0.1, 0.9, 0.3, 0.2, 0.0],
                        [0.4, 0.5, 0.6, 0.4, 0.0],
                        [0.4, 0.3, 0.7, 0.1, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    r = gumbel_sinkhorn(p, 0.1)
    row_sums = torch.sum(r, dim=1)
    col_sums = torch.sum(r, dim=0)
    print(row_sums)
    print(col_sums)
    print(r)

# ------------------------------------------------------------------------
def pad_tensor():
    p = torch.tensor([[0.1, 0.9, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.4, 0.3, 0.7]], dtype=torch.float32)

    r = F.pad(p,pad=(0,1,0,2)) # 左右上下
    print(r)

# ------------------------------------------------------------------------
def generate_line_graph():
    edge_index = [[7, 3], [3, 1], [3, 8], [3, 9], [5, 2], [6, 2], [1, 0], [0, 2], [4, 2]]
    features = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    features = torch.tensor(features).float()
    print(features.shape)
    G = nx.Graph()
    # 生成线图边索引
    edge_index = [tuple(i) for i in edge_index]    # 元素转换为元组
    G.add_edges_from(edge_index)
    L = nx.line_graph(G)  # 生成线图L(G)
    edge_index_2 = L.edges()  # 得到线图的边索引
    # 生成线图节点特征向量
    feature_list = []
    for node in L.nodes():
        f = features[node[0]] + features[node[1]]
        feature_list.append(f.unsqueeze(0))
    features_2 = torch.cat(feature_list, dim=0)  
    print(features_2.shape)
    node_mapping = {edge: idx for idx, edge in enumerate(L.nodes())}
    L_mapped = nx.Graph()
    for edge in L.edges():
        # 获取映射后的节点索引
        new_u = node_mapping[edge[0]]
        new_v = node_mapping[edge[1]]
        # 在新图中添加边
        L_mapped.add_edge(new_u, new_v)
    print("原边图的节点:", list(L.nodes()))
    print("映射后的节点:", node_mapping)
    print("原边图的边:", L.edges())
    print("新边图的边:", L_mapped.edges())

    # 绘制原图G
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    nx.draw(G, with_labels=True, node_color='orange', edge_color='black', node_size=1000, font_size=16)
    plt.title('Original Graph G')
    # 绘制线图L(G)
    plt.subplot(122)
    nx.draw(L, with_labels=True, node_color='lightblue', edge_color='black', node_size=1000, font_size=16)
    plt.title('Line Graph L(G)')
    plt.show()

def transform_to_line_graph():
    edge_index = torch.tensor([
        [7, 7, 3, 5, 8, 6, 1, 1, 0, 6, 9, 0, 2, 6, 4, 0, 4, 2, 0, 1, 2, 3, 4, 5,
         6, 7, 8, 9],
        [6, 9, 0, 2, 6, 4, 0, 4, 2, 7, 7, 3, 5, 8, 6, 1, 1, 0, 0, 1, 2, 3, 4, 5,
         6, 7, 8, 9]])
    # 去除原图边索引中的自环和反向对
    _, n = edge_index.shape
    x = edge_index[0, -1].item()
    edge_index = edge_index[:, :n-x-1]
    _, n = edge_index.shape
    edge_index = edge_index[:, :int(n/2)]
    print(edge_index)

def node_alignment_with_edge():
    map_matrix = torch.tensor([
        [-0.0876, -0.0401, -0.0679, -0.0539,  0.0392, -0.2828, -0.2195, -0.0533, -0.1780, -0.0579],
        [-0.0534, -0.1842, -0.1351, -0.1889, -0.1086, -0.2122, -0.1998, -0.1768, -0.1016, -0.1773],
        [-0.1114, -0.1309, -0.1408, -0.0850, -0.0772, -0.2847, -0.2259, -0.2473, -0.2625, -0.2388],
        [-0.1479, -0.1357, -0.2550, -0.1085, -0.2229, -0.1593, -0.2189, -0.0714, -0.2016, -0.1402],
        [-0.1510, -0.1799, -0.1988, -0.2105, -0.1961, -0.1336, -0.2262, -0.2012, -0.1765, -0.2066],
        [-0.1833, -0.2587, -0.1745, -0.1439, -0.1256, -0.1921, -0.2084, -0.0687, -0.2650, -0.1643],
        [-0.1831, -0.2106, -0.1560, -0.2474, -0.2259, -0.1658, -0.1147, -0.1671, -0.1263, -0.1683],
        [-0.2363, -0.2498, -0.0784, -0.2131, -0.1367, -0.1598, -0.1704, -0.1169, -0.1856, -0.1105],
        [-0.1901, -0.3069,  0.0068, -0.3052, -0.2435, -0.0969, -0.0535, -0.1415, -0.1045, -0.1632],
        [-0.2426, -0.3135,  0.0143, -0.2249, -0.0932, -0.1302, -0.1505, -0.1765, -0.1094, -0.1907]])
    lg_map_matrix = torch.tensor([
        [-0.0174, -0.0569,  0.0153, -0.0872, -0.0328,  0.0371,  0.0961,  0.0277,  0.0971,  0.0274],
        [ 0.0112, -0.0126,  0.0204,  0.1814,  0.1416,  0.0743,  0.1157, -0.0636, -0.0100, -0.0223],
        [-0.0380,  0.0420,  0.0623,  0.0448, -0.0204, -0.0494, -0.0158,  0.0099,  0.0199,  0.0354],
        [ 0.0390, -0.0507, -0.0554,  0.0489, -0.0484, -0.0530,  0.0103, -0.0065, -0.0228, -0.0496],
        [ 0.0981,  0.0433, -0.0742,  0.0914, -0.0243,  0.0038, -0.0543, -0.0250, -0.0089, -0.0010],
        [-0.0235, -0.0165,  0.0691, -0.0567, -0.0114,  0.0080, -0.0235, -0.0239, -0.0155, -0.0181],
        [-0.0289,  0.0035, -0.0166, -0.0152,  0.0134, -0.0119,  0.0633,  0.0041,  0.0342,  0.0003],
        [-0.0104,  0.0826,  0.0475,  0.2230, -0.0193,  0.0216,  0.1876,  0.0101,  0.1135, -0.0113],
        [-0.0288, -0.0163, -0.0098,  0.0707,  0.1436, -0.0418,  0.2567,  0.0139, -0.0300,  0.0750]])
    lg_node_list_1 = [(7, 9), (3, 0), (5, 2), (6, 8), (0, 2), (6, 4), (7, 6), (0, 1), (4, 1)]
    lg_node_list_2 = [(3, 1), (2, 4), (2, 6), (1, 0), (2, 0), (5, 2), (7, 5), (7, 9), (9, 8), (8, 6)]
    print("节点相似度矩阵:", map_matrix.shape)
    print("边相似度矩阵:", lg_map_matrix.shape)
    print("-------------------------------")
    # print(len(lg_node_list_1))
    # print(len(lg_node_list_2))
    aligment_index = []
    n, m = map_matrix.shape
    x = min(n, m)  # 最多对齐的节点数
    i = 0
    while i < x:
        """ 1、从节点相似度矩阵中得到节点对x """
        max_index = torch.argmax(map_matrix).item()
        max_row = max_index // m
        max_col = max_index % m
        print("原图节点对x:", max_row, max_col)
        aligment_index.append([max_row, max_col])
        i += 1
        if i == x: break
        map_matrix[max_row, :] = -1
        map_matrix[:, max_col] = -1
        """ 2、根据节点对x, 从边相似度矩阵找到另一节点对y """
        row_list = []  # 在边相似度矩阵中，与得到“原图节点”索引相关的“边图节点”索引
        col_list = []
        for index in range(len(lg_node_list_1)):
            if max_row in lg_node_list_1[index]: row_list.append(index)
        for index in range(len(lg_node_list_2)):
            if max_col in lg_node_list_2[index]: col_list.append(index)
        # print("row_list:", row_list)
        # print("col_list:", col_list)
        # 得到边图节点对z
        max_score = -1
        lg_node_1, lg_node_2 = (), ()
        for j in row_list:  # 遍历边相似度度矩阵中“原图节点”相关的元素，选择最大值
            for k in col_list:
                if lg_map_matrix[j][k] > max_score:
                    max_score = lg_map_matrix[j][k]  # 更新最大值
                    lg_node_1 = lg_node_list_1[j]  # 更新索引
                    lg_node_2 = lg_node_list_2[k]
        print("边图节点对z:", lg_node_1, lg_node_2)
        # 从边图节点对z中得到原图节点对y
        new_row = lg_node_1[0] if max_row == lg_node_1[1] else lg_node_1[1]  
        new_col = lg_node_2[0] if max_col == lg_node_2[1] else lg_node_2[1]
        print("原图节点对y:", new_row, new_col)
        """ 3、根据节点相似度矩阵的情况, 判断节点对y是否舍弃 """
        if map_matrix[new_row, new_col] != -1:
            aligment_index.append([new_row, new_col])
            i += 1
            map_matrix[new_row, :] = -1
            map_matrix[:, new_col] = -1
            print("1")
        print("-----------------------")
        
    map_matrix.zero_()
    for index in aligment_index:
        map_matrix[index[0], index[1]] = 1
    print(map_matrix)
    return map_matrix

def Euclidean_Distance():
    # 假设 embeddings1 是 n x 32 的张量，embeddings2 是 m x 32 的张量
    n, d = 5, 32  # 例如 n = 5
    m = 7  # 例如 m = 7
    embeddings1 = torch.randn(n, d)
    embeddings2 = torch.randn(m, d)

    # 扩展两个张量以进行向量化距离计算
    # embeddings1.unsqueeze(1) 将变成 n x 1 x 32
    # embeddings2.unsqueeze(0) 将变成 1 x m x 32
    # 结果将会广播成 n x m x 32
    diff = embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0)

    # 计算欧氏距离：先平方，然后在最后一个维度上求和，最后开根号
    dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=2))
    # 计算最小值和最大值
    min_val = torch.min(dist_matrix)
    max_val = torch.max(dist_matrix)
    # 进行最小-最大归一化
    normalized_dist_matrix = (dist_matrix - min_val) / (max_val - min_val)
    normalized_dist_matrix = normalized_dist_matrix * 2 -1

    print("Cost matrix (n x m):\n", normalized_dist_matrix)

def f1():
    """ 节点相似度矩阵和边相似度矩阵得到一个有偏向性的噪声代替gs中的随机噪声"""
    map_matrix = torch.tensor([
        [-0.0876, -0.0401, -0.0679, -0.0539,  0.0392, -0.2828, -0.2195, -0.0533, -0.1780, -0.0579],
        [-0.0534, -0.1842, -0.1351, -0.1889, -0.1086, -0.2122, -0.1998, -0.1768, -0.1016, -0.1773],
        [-0.1114, -0.1309, -0.1408, -0.0850, -0.0772, -0.2847, -0.2259, -0.2473, -0.2625, -0.2388],
        [-0.1479, -0.1357, -0.2550, -0.1085, -0.2229, -0.1593, -0.2189, -0.0714, -0.2016, -0.1402],
        [-0.1510, -0.1799, -0.1988, -0.2105, -0.1961, -0.1336, -0.2262, -0.2012, -0.1765, -0.2066],
        [-0.1833, -0.2587, -0.1745, -0.1439, -0.1256, -0.1921, -0.2084, -0.0687, -0.2650, -0.1643],
        [-0.1831, -0.2106, -0.1560, -0.2474, -0.2259, -0.1658, -0.1147, -0.1671, -0.1263, -0.1683],
        [-0.2363, -0.2498, -0.0784, -0.2131, -0.1367, -0.1598, -0.1704, -0.1169, -0.1856, -0.1105],
        [-0.1901, -0.3069,  0.0068, -0.3052, -0.2435, -0.0969, -0.0535, -0.1415, -0.1045, -0.1632],
        [-0.2426, -0.3135,  0.0143, -0.2249, -0.0932, -0.1302, -0.1505, -0.1765, -0.1094, -0.1907]])
    aligment_matrix = node_alignment_with_edge()
    print("++++++++++++++++++++++++++++++")
    # print("uniform_noise:\n", uniform_noise)
    for i in range(5):
        uniform_noise = torch.rand_like(map_matrix)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        # print("gumbel_noise:\n", gumbel_noise)
        biased_gumbel_noise = torch.mul(gumbel_noise, (1 + aligment_matrix * 0.5))
        # print("biased_gumbel_noise:\n", biased_gumbel_noise)
        a1 = (map_matrix + gumbel_noise)/0.1
        a2 = (map_matrix + biased_gumbel_noise)/0.1
        result1 = log_sinkhorn_norm(a1, 20)
        result2 = log_sinkhorn_norm(a2, 20)
        # print("row_sums:", torch.sum(result1, dim=1))
        # print("col_sums:", torch.sum(result1, dim=0))
        # print("result:\n", result)
        print(torch.sum(torch.mul(aligment_matrix, result1)))
        # print("row_sums:", torch.sum(result2, dim=1))
        # print("col_sums:", torch.sum(result2, dim=0))
        print(torch.sum(torch.mul(aligment_matrix, result2)))
        print("--------------------------------")

if __name__ == "__main__":
    # node_alignment_with_edge()
    # transform_to_line_graph()
    # ----------------------------------
    f1()
    pass