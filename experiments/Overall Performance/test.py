import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

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
now = '_'+datetime.now().strftime('%y-%m%d-%H%M')+'.txt'

print(now)