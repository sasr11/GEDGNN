import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import random
import networkx as nx
import matplotlib.pyplot as plt

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

# logits1 = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]], dtype=torch.float32)
# logits2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32)
# logits3 = torch.tensor([0.11, 0.22, 0.33, 0.44, 0.55], dtype=torch.float32)

# print(torch.cat((logits2.repeat(3, 1), logits3.repeat(3, 1)), dim=1))

# print(logits1.repeat(3, 1))
# print(logits.size())
# print(logits1.squeeze())
# print(logits.squeeze(-1).size())
# for i in range(5):
#     print(gumbel_softmax_test(logits, 5))
#     print("---------------------")

# for i in range(5):
#     print(gumbel_softmax(logits))
#     print("---------------------")

# --------------------------------------------------------------------------------------

# 示例：创建一个n*m的张量x
# n, m = 5, 4  # 假设张量x的形状为5x4
# x = torch.arange(n*m).view(n, m)  # 创建一个示例张量
# print("原始张量x:\n", x)
# rows, cols = x.shape[:2]
# print(rows, cols)
# print(x.numel())
# print(x.view(1, x.numel()))
# 索引张量，形状为n*1，这里用一个示例，表示要归零第2和第4行（索引从0开始）
# index = torch.tensor([0, 1, 0, 1])
# print(index)
# 将索引张量转换为布尔张量
# mask = index == 1
# mask = mask.expand_as(x)
# print(mask)
# 使用布尔索引归零对应的行

# print("\n修改后的张量x:\n", x[mask])
# x[mask.expand_as(x)] = 0
# print("\n修改后的张量x:\n", x)

#---------------------------------------------------------------------------------------

# p = torch.tensor([[0, 1, 0],
#                   [1, 0, 0],
#                   [0, 0, -1]], dtype=torch.float32)
# t = torch.tensor([[1, 0, 0],
#                   [0, 1, 0],
#                   [0, 0, 1]], dtype=torch.float32)
# loss = torch.nn.BCEWithLogitsLoss()
# print(loss(p, t))


# -------------------------------------------------------------------------------------------

# 获取当前日期和时间
# now = '_'+datetime.now().strftime('%y-%m%d-%H%M')+'.txt'

# print(now)

# -------------------------------------------------------------------------------------------

# def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int =20):
#     for _ in range(n_iter):
#         log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
#         log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
#     return log_alpha.exp()

# def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 1.0, n_iter: int = 20, noise: bool = True):
#     if noise:
#         uniform_noise = torch.rand_like(log_alpha)
#         gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
#         log_alpha = (log_alpha + gumbel_noise)/tau
#     sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
#     return sampled_perm_mat

# p = torch.tensor([[0.1, 0.9, 0.3, 0.2],
#                     [0.4, 0.5, 0.6, 0.3],
#                     [0.4, 0.3, 0.7, 0.1],
#                     [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
# r = gumbel_sinkhorn(p, 0.1)
# row_sums = torch.sum(r, dim=1)
# col_sums = torch.sum(r, dim=0)
# print(row_sums)
# print(col_sums)
# print(r)

# ------------------------------------------------------------------------
# p = torch.tensor([[0.1, 0.9, 0.3],
#                     [0.4, 0.5, 0.6],
#                     [0.4, 0.3, 0.7]], dtype=torch.float32)

# r = F.pad(p,pad=(0,1,0,2)) # 左右上下
# print(r)
# ---------------------------------------------------------------

# 创建一个图G
# G = nx.Graph()
# G.add_edges_from([('1', '2'), ('1', '3'), ('1', '4'), ('2', '5'), ('4', '3'), ('4', '5')])
# # 绘制原图G
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# nx.draw(G, with_labels=True, node_color='orange', edge_color='black', node_size=1000, font_size=16)
# plt.title('Original Graph G')

# # 生成线图L(G)
# L = nx.line_graph(G)

# # 绘制线图L(G)
# plt.subplot(122)
# nx.draw(L, with_labels=True, node_color='lightblue', edge_color='black', node_size=1000, font_size=16)
# plt.title('Line Graph L(G)')

# plt.show()

# ------------------------------------------------------------------------
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
print(L.edges())
print("------------------------------------------")
print(L.nodes())
print("------------------------------------------")
print(features_2.shape)

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