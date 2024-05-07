import torch
import random
import time
import networkx as nx

def my_mask(masked_index, mapping):
        """
        根据masked_index遮蔽mapping中的行或列
        :param masked_index: 要遮蔽的行或列索引 1*n
        :param mapping: 要进行遮蔽的矩阵 
        :return mapping: 遮蔽后的矩阵
        """ 
        if masked_index == None: return mapping
        rows, cols = mapping.shape[:2]
        num = masked_index.numel()
        if rows < cols:  # G1节点数小于G2节点数
            if (num == cols):  # 判断节点索引与节点数相同
                masked_index = masked_index.view(1, num)  # 使张量形状变为1*num
                mask = masked_index == 0  # 将索引张量转换为布尔张量
                mapping[mask.expand_as(mapping)] = 0  # expand_as将mask拓展与pre_mapping一样的形状
                return mapping
            else: assert False
        elif rows > cols:  # G1节点数大于G2节点数
            if (num == rows):  # 判断节点索引与节点数相同
                masked_index = masked_index.view(num, 1)  # 使张量形状变为num*1
                mask = masked_index == 0  # 将索引张量转换为布尔张量
                mapping[mask.expand_as(mapping)] = 0  # expand_as将mask拓展与pre_mapping一样的形状
                return mapping
            else: assert False
    
def my_alignment(masked_index, mapping):
    """
    根据masked_index遮蔽mapping中的行或列后, 对剩余节点进行对齐(变成0-1矩阵)
    :param masked_index: 要遮蔽的行或列索引 1*n
    :param mapping: 要进行对齐的矩阵 
    :return mapping: 对齐后的矩阵
    """ 
    mapping = my_mask(masked_index, mapping)
    n, m = mapping.shape
    x = min(n, m)
    aligment_index = []
    for _ in range(x):
        # 找到最大元素的索引
        max_index = torch.argmax(mapping)
        max_row = max_index // m
        max_col = max_index % m
        aligment_index.append([max_row, max_col])
        # 设置最大元素所在行列的值为-1
        mapping[max_row, :] = -1
        mapping[:, max_col] = -1
    
    mapping.zero_()
    for index in aligment_index:
        mapping[index[0]][index[1]] = 1
    return mapping

def my_match(mapping, gt_mapping):
    # 预测的对齐结果与真实对齐结果的数量
    sum = 0
    for i, j in zip(mapping.reshape(-1), gt_mapping.reshape(-1)):
        if(i==1 and j==1): sum = sum + 1
    return sum

def my_pad_features(f1: torch.Tensor, f2: torch.Tensor, n1: int, n2: int):
    """通过填充节点one-hot向量的方式添加节点，使图对节点数相同
    Args:
        f1 (torch.Tensor): 图1节点特征 n1*29
        f2 (torch.Tensor): 图2节点特征 n2*29
        n1 (int): 图1节点数量
        n2 (int): 图2节点数量
    """
    d = f1.shape[1]
    if n1 > n2:
        for i in range(n1-n2):
            # 创建一个1xd的全零向量
            padded_features = torch.zeros(1, d)
            # random_index = torch.randint(0, d, (1,))
            # padded_features[0, random_index] = 1
            f2 = torch.cat((f2, padded_features), dim=0)
        return f1, f2, n1, n1
    elif n1 < n2:
        for i in range(n2-n1):
            # 创建一个1xd的全零向量
            padded_features = torch.zeros(1, d)
            # random_index = torch.randint(0, d, (1,))
            # padded_features[0, random_index] = 1
            f1 = torch.cat((f1, padded_features), dim=0)
        return f1, f2, n2, n2
    
def my_lineGraph(edge_index: list, features: torch.Tensor):
    """将原图转换为边图, 并生成边图的节点特征向量
    Args:
        edge_index (list): 原图的边索引 [(1,2),(2,4)...]
        features (torch.Tensor): 原图的节点特征向量N*d tensor([[],[],...])
    Returns:
        lg_node_list: 边图的节点列表
        lg_edge_index_mapping: 边图的边索引映射 
        lg_features: 边图的节点特征向量
        L.number_of_nodes(): 边图的节点数
    """
    # 去除原图边索引中的自环和反向对
    # print("原图边索引:", edge_index)  # zhj
    _, n = edge_index.shape
    x = edge_index[0, -1].item()
    edge_index = edge_index[:, :n-x-1]  # 去除自环
    _, n = edge_index.shape
    edge_index = edge_index[:, :int(n/2)]  # 去除反向对
    # print("去除自环和反向对的原图边索引:", edge_index)  # zhj
    # 生成线图边索引
    G = nx.Graph()
    edge_index = [tuple([i.item(),j.item()]) for i,j in zip(edge_index[0],edge_index[1])]  # 2*n(Torch.tensor) -> n*2(list)
    G.add_edges_from(edge_index)
    L = nx.line_graph(G)  # 生成线图L(G)
    # 生成边图的节点特征向量
    feature_list = []
    for node in L.nodes():
        f = features[node[0]] + features[node[1]]  # 两个端点的特征向量相加
        feature_list.append(f.unsqueeze(0))
    lg_features = torch.cat(feature_list, dim=0)
    # 建立L的节点（边的元组）到单个整数的映射, 得到在新的节点映射下的边索引（后续的图卷积网络需要）
    lg_edge_index_mapping = []
    lg_node_mapping = {edge: idx for idx, edge in enumerate(L.nodes())}
    for edge in L.edges():
        lg_edge_index_mapping.append([lg_node_mapping[edge[0]], lg_node_mapping[edge[1]]])
    lg_edge_index_mapping += [[y, x] for x, y in lg_edge_index_mapping]  # 添加反向对
    lg_edge_index_mapping += [[x, x] for x in range(L.number_of_nodes())]  # 添加自环
    lg_edge_index_mapping = torch.tensor(lg_edge_index_mapping).t().long()  # n*2(list) -> 2*n(torch.Tensor)
    return list(L.nodes()), lg_edge_index_mapping, lg_features, L.number_of_nodes()