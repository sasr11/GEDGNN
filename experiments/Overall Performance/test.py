import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.glob import global_add_pool
import numpy as np
from datetime import datetime
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import ast
import os
import json
from matplotlib import cm


def load_data():
    with open('./s.txt', 'r') as f:  # 从文件中读取
        matrix_pairs_loaded = []
        num = 1
        t_list = [7, 8]  # 想要提取第几行数据
        for line in f:
            if num in t_list:
                pair = ast.literal_eval(line.strip())  # 使用 literal_eval 将字符串转换为 Python 对象
                # gid1, gid2, ged, pre_score, similarity_matrix, alignment_matrix
                matrix_pairs_loaded.append((pair[0], pair[1], pair[2], pair[3], np.array(pair[4]), np.array(pair[5])))
            num += 1
            # if count > 0: break
    return matrix_pairs_loaded

def load_data2():
    count1 = 0  # GED小于5
    count2 = 0  # GED小于10
    count3 = 0  # GED大于等于10
    count4 = 0  # sim小于0.2
    count5 = 0  # sim小于0.5
    count6 = 0  # sim大于等于0.5
    sum1 = 0
    sum2 = 0
    with open('./save.txt', 'r') as f:  # 从文件中读取
        num = 1
        for line in f:
            # gid1, gid2, ged, pre_score, similarity_matrix, alignment_matrix
            pair = ast.literal_eval(line.strip())  # 使用 literal_eval 将字符串转换为 Python 对象
            ged = pair[2]
            sim = pair[3]
            if ged < 5: count1 += 1
            elif ged < 10: count2 += 1
            else: count3 += 1
            if sim < 0.2: count4 += 1
            elif sim < 0.5: count5 += 1
            else: count6 += 1
            if num % 10000 == 1:  print(num)
            num += 1
    sum1 = count1 + count2 + count3
    sum2 = count4 + count5 + count6
    print(count1, count2, count3, sum1)  # 2490 49696 36224 88410
    print(count4, count5, count6, sum2)  # 651 53868 33891 88410
        

def draw(data):
    # 绘制矩阵，根据数值大小显示颜色深浅
    # data [(gid1, gid2, ged, pre_score, similarity_matrix, alignment_matrix), ...]
    
    num = 0
    for pair in data:
        file1 = './json_data/AIDS/train/{}.json'.format(pair[0])  # 根据gid读取图相关数据
        file2 = './json_data/AIDS/train/{}.json'.format(pair[1])
        g1 = json.load(open(file1, 'r'))
        g2 = json.load(open(file2, 'r'))
        n1 = g1["n"]
        n2 = g2["n"]
        sim = pair[4]  # 相似度矩阵
        sim = sim[:n1, :n2]  
        align = pair[5]  # 对齐矩阵
        align = align[:n1, :n2] 
        align = np.where(align > 0.5, 1.0, 0.0)  
        count = np.sum(align)
        if count < align.shape[0]:  # 如果没有全部对齐，跳过
            num += 1
            continue
        # ----------------------------------------------------------------
        plt.matshow(sim, cmap=plt.get_cmap('Greens'), alpha=0.6)
        rows, cols = align.shape
        for i in range(rows):
            for j in range(cols):
                if align[i, j] == 1:
                    plt.plot(j, i, 'rx')  # j 对应列，i 对应行，'ro' 表示红色圆圈
        plt.xticks(np.arange(0, cols, 1))
        plt.yticks(np.arange(0, rows, 1))
        plt.savefig("./pictures/{}_1.png".format(num))
        print(num)
        draw2(g1, g2, align, num)
        num += 1

def draw2(g1, g2, align, num):
    # 绘制图对齐
    n1 = g1["n"]
    n2 = g2["n"]
    # 示例的两个分子图
    G1 = nx.Graph()
    G2 = nx.Graph()
    # 添加节点和边 (分子图中的原子和键)
    G1.add_edges_from(g1["graph"]) 
    G2.add_edges_from(g2["graph"]) 
    align = align[:n1, :n2]
    atoms_G1 = g1["labels"]  # G1 的节点的原子类型
    atoms_G2 = g2["labels"]  # G2 的节点的原子类型
    
    # 为不同原子类型指定不同颜色
    atom_types = list(set(atoms_G1 + atoms_G2))
    # atom_types = [
    #     "As", "B", "Bi", "Br", "C", "Cl", "Co", "Cu", "F", "Ga", 
    #     "Hg", "Ho", "I", "Li", "N", "Ni", "O", "P", "Pb", "Pd", 
    #     "Pt", "Ru", "S", "Sb", "Se", "Si", "Sn", "Tb", "Te"]
    
    # 创建颜色映射
    cmap = [
        '#FF5733', '#33FF57', '#3357FF', '#F4FF33', '#33FFF6', '#FF33A2', '#FF9A33',
        '#C433FF', '#8BFF33', '#33FFBD', '#FF5A33', '#57FF33', '#FF338A', '#33D4FF',
        '#FFB433', '#FF33D2', '#B6FF33', '#334EFF', '#FF3382', '#6CFF33', '#33FF6E',
        '#33B7FF', '#FF339A', '#5EFF33', '#FF3342', '#33FFF3', '#FF335A', '#89FF33', 
        '#7A33FF'
    ]
    cmap = cmap[:len(atom_types)]
    # cmap = cm.get_cmap('tab20', len(atom_types))  # 'tab20' colormap with 29 unique colors
    # 为每个原子类型生成颜色映射
    atom_color_map = {atom: cmap[i] for i, atom in enumerate(atom_types)}
    # 根据原子类型为节点上色
    colors_G1 = [atom_color_map[atom] for atom in atoms_G1]
    colors_G1 = [colors_G1[node] for node in G1.nodes()]  # 由于节点顺序是按照创建顺序来的，所以得重新对应颜色
    colors_G2 = [atom_color_map[atom] for atom in atoms_G2]
    colors_G2 = [colors_G2[node] for node in G2.nodes()]
    
    # 创建图的布局
    pos_G1 = nx.spring_layout(G1)
    pos_G2 = nx.spring_layout(G2)

    # 偏移 G2 的节点位置，方便可视化
    offset = 2.5
    pos_G2_offset = {node: (x + offset, y) for node, (x, y) in pos_G2.items()}

    # 创建画布
    plt.figure(figsize=(10, 5))
    
    # 绘制图 G1，标注原子类型
    nx.draw(G1, pos_G1, with_labels=True, labels={i: atoms_G1[i] for i in range(len(atoms_G1))}, 
            node_color=colors_G1, edge_color='gray', node_size=500)
    plt.title('Graph G1 and G2 with Alignment')

    # 绘制图 G2，标注原子类型
    nx.draw(G2, pos_G2_offset, with_labels=True, labels={i: atoms_G2[i] for i in range(len(atoms_G2))}, 
            node_color=colors_G2, edge_color='gray', node_size=500)

    # 绘制对齐关系
    for i in range(align.shape[0]):  # 遍历 G1 的节点
        for j in range(align.shape[1]):  # 遍历 G2 的节点
            if align[i, j] == 1:  # 如果对齐矩阵中对应位置为 1
                x_values = [pos_G1[i][0], pos_G2_offset[j][0]]  # 获取 G1 和 G2 中对应节点的 x 坐标
                y_values = [pos_G1[i][1], pos_G2_offset[j][1]]  # 获取 G1 和 G2 中对应节点的 y 坐标
                plt.plot(x_values, y_values, 'r--')  # 使用红色虚线连接对齐的节点
    plt.savefig("./pictures/{}_2.png".format(num))

def f():

    # 示例的两个分子图 G1 和 G2
    G1 = nx.Graph()

    # 添加节点和边 (分子图中的原子和键)
    G1.add_edges_from([[7, 4], [7, 9], [3, 1], [3, 6], [5, 8], [5, 2], [8, 9], [6, 4], [1, 0], [1, 2], [4, 2]])  # G1 四边形

    # 创建图的布局
    pos_G1 = nx.spring_layout(G1)
    
    colors = ['red', 'blue', 'blue', 'red', 'blue', 'blue', 'red', 'blue', 'blue', 'blue']
    
    # colors_dict = {0: 'red', 1: 'blue', 2: 'blue', 3: 'red', 4: 'blue', 5: 'blue', 6: 'red', 7: 'blue', 8: 'blue', 9: 'blue'}
    colors = [colors[node] for node in G1.nodes()]

    # 创建画布
    plt.figure(figsize=(10, 5))

    # 绘制图 G1
    nx.draw(G1, pos_G1, with_labels=True, node_color=colors, edge_color='gray', node_size=500)


    plt.savefig("./pictures/x.png")


if __name__ == "__main__":
    # 在“GEDGNN/experiments/Overall Performance”目录下运行
    data = load_data()
    draw(data)
    # f()
    # load_data2()
    pass

    