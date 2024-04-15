import torch
import random
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
            f2 = torch.cat((f2, padded_features), dim=0)
        return f1, f2, n1, n1
    elif n1 < n2:
        for i in range(n2-n1):
            # 创建一个1xd的全零向量
            padded_features = torch.zeros(1, d)
            f1 = torch.cat((f1, padded_features), dim=0)
        return f1, f2, n2, n2
    
def my_lineGraph(edge_index: list, features: torch.Tensor):
    """将原图转换为边图, 并生成边图的节点特征向量
    Args:
        edge_index (list): 原图的边索引 [(1,2),(2,4)...]
        features (torch.Tensor): 原图的节点特征向量N*d tensor([[],[],...])
    Returns:
        lg_edge_index: 边图的边索引
        lg_features: 边图的节点特征向量
        L.number_of_nodes(): 边图的节点数
    """
    G = nx.Graph()
    # 生成线图边索引
    edge_index = [tuple(i) for i in edge_index]  # 元素转换为元组
    G.add_edges_from(edge_index)
    L = nx.line_graph(G)  # 生成线图L(G)
    lg_edge_index = L.edges()  # 得到线图的边索引
    # 生成边图的节点特征向量
    feature_list = []
    for node in L.nodes():
        f = features[node[0]] + features[node[1]]  # 两个端点的特征向量相加
        feature_list.append(f.unsqueeze(0))
    lg_features = torch.cat(feature_list, dim=0)
    return lg_edge_index, lg_features, L.number_of_nodes()