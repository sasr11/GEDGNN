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
        for line in f:
            pair = ast.literal_eval(line.strip())  # 使用 literal_eval 将字符串转换为 Python 对象
            if pair[2] < 7 and pair[3] < 0.4:
                # gid1, gid2, ged, pre_score, similarity_matrix, alignment_matrix
                matrix_pairs_loaded.append((pair[0], pair[1], pair[2], pair[3], np.array(pair[4]), np.array(pair[5]), num))
            num += 1
            # if count > 0: break
    return matrix_pairs_loaded

def load_rank_data(model, dataset):
    if dataset == "AIDS": batch = 420
    else: batch = 600
    with open('./rank_{}_{}.txt'.format(model, dataset), 'r') as f:  # 从文件中读取
        data_loaded = []
        gid_list = []
        ged_list = []
        pre_list = []
        nged_list = []
        num = 1
        for line in f:
            # gid1, gid2, ged, pre_ged, nged
            pair = ast.literal_eval(line.strip())  # 使用 literal_eval 将字符串转换为 Python 对象            
            gid_list.append(pair[1])
            ged_list.append(pair[2])
            pre_list.append(pair[3])
            nged_list.append(pair[4])
            if num % batch == 0:
                data_loaded.append((pair[0], gid_list, ged_list, pre_list, nged_list))
                gid_list = []
                ged_list = []
                pre_list = []
                nged_list = []
            num += 1
            # if count > 0: break
    return data_loaded

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
        
def draw_case(data):
    # 绘制矩阵，根据数值大小显示颜色深浅
    # data [(gid1, gid2, ged, pre_score, similarity_matrix, alignment_matrix), ...]
    
    num = 0  # 统计没有全部对齐的数量
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
        # count = np.sum(align)
        # if count < align.shape[0]:  # 如果没有全部对齐，跳过
        #     num += 1
        #     continue
        # ----------------------------------------------------------------
        print(pair[6])  # 输出序号
        draw1(sim, align, pair[6])
        draw2(g1, g2, align, pair[6])  # 绘制图对齐
    print("{}/{}".format(num, len(data)))

def draw1(sim, align, num):
    # 绘制相似度矩阵
    plt.matshow(sim, cmap=plt.get_cmap('Greens'), alpha=0.6)
    rows, cols = align.shape
    for i in range(rows):
        for j in range(cols):
            if align[i, j] == 1:
                plt.plot(j, i, 'rx')  # j 对应列，i 对应行，'ro' 表示红色圆圈
    plt.xticks(np.arange(0, cols, 1))
    plt.yticks(np.arange(0, rows, 1))
    plt.savefig("./pictures/{}_1.png".format(num))

def draw2(g1, g2, align, num):
    # 绘制图对齐结果
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

def draw3(g1, gid, num):
    """绘制单个图
    Args:
        g1 (_type_): 图相关数据：节点数n，边数m，标签列表labels，边列表graph
        gid (_type_): 图id，与文件相同
        num (_type_): 序号，rank专用
    """
    G1 = nx.Graph()
    # 添加节点和边 (分子图中的原子和键)
    G1.add_edges_from(g1["graph"]) 
    atoms_G1 = g1["labels"]  # G1 的节点的原子类型
    
    # 为不同原子类型指定不同颜色
    atom_types = list(set(atoms_G1))
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
    
    # 创建图的布局
    pos_G1 = nx.spring_layout(G1)

    # 创建画布
    plt.figure(figsize=(10, 5))
    plt.title('Graph')
    # 绘制图 G1，标注原子类型
    nx.draw(G1, pos_G1, with_labels=True, labels={i: atoms_G1[i] for i in range(len(atoms_G1))}, 
            node_color=colors_G1, edge_color='gray', node_size=500)
    plt.savefig("./pictures/rank/AIDS/rank_{}_{}.png".format(num, gid))

def draw_aids_rank(g_list, gid_list, nged_list, model, dataset):
    """绘制rank图, AIDS

    Args:
        g_list (_type_): 要绘制的图列表
        gid_list (_type_): 图ID列表
        model (_type_): 模型名称
        dataset (_type_): 数据集名称
    """
    num_graphs = len(g_list)  # 图的数量
    fig, axs = plt.subplots(1, num_graphs, figsize=(36, 5))  # 创建子图，按行排列
    graphs = []
    atom_all = []
    atoms_list = []
    for g in g_list:
        G = nx.Graph()
        G.add_edges_from(g["graph"])
        atom_all.extend(g["labels"])
        atoms_list.append(g["labels"]) 
        graphs.append(G)
    # 为每个原子类型生成颜色映射
    atom_color_map = {
     "As": '#FF335A', "B": '#33FFF3', "Bi": '#3357FF', "Br": '#F4FF33', "C": '#33FFF6', 
     "Cl": '#FF33A2', "Co": '#FF9A33', "Cu": '#C433FF', "F": '#8BFF33', "Ga": '#33FFBD', 
     "Hg": '#FF5A33', "Ho": '#57FF33', "I": '#FF338A', "Li": '#33D4FF', "N": '#FFB433', 
     "Ni": '#FF33D2', "O": '#B6FF33', "P": '#334EFF', "Pb": '#FF3382', "Pd": '#6CFF33', 
     "Pt": '#33FF6E', "Ru": '#33B7FF', "S": '#FF339A', "Sb": '#5EFF33', "Se": '#FF3342', 
     "Si": '#33FF57', "Sn": '#FF5733', "Tb": '#89FF33', "Te": '#7A33FF'}
    titles = ['Rank by ' + model]
    titles.extend(nged_list)
    subtitles = ['', 1,2,3,4,5,6,21,420]
    # 遍历每个图并绘制在相应的子图中
    for i, g in enumerate(graphs):
        ax = axs[i]  # 选择对应的子图
        colors = [atom_color_map[atom] for atom in atoms_list[i]]
        colors = [colors[node] for node in g.nodes()]  # 由于节点顺序是按照创建顺序来的，所以得重新对应颜色
        # 如果没有指定位置，则使用 spring 布局
        pos = nx.spring_layout(g, seed=42)  # 固定随机种子的布局
        # 绘制节点和边
        nx.draw(g, pos, with_labels=False, 
                ax=ax, node_color=colors, node_size=500, edge_color='black')
        ax.set_title(titles[i], fontsize=18)
        ax.text(0.5, -0.1, subtitles[i], fontsize=16, ha='center', transform=ax.transAxes)
    plt.savefig("./pictures/rank/AIDS/rank_{}_{}_{}.png".format(model, dataset, gid_list[0]))

def draw_linux_rank(g_list, gid_list, nged_list, model, dataset):
    num_graphs = len(g_list)  # 图的数量
    fig, axs = plt.subplots(1, num_graphs, figsize=(36, 5))  # 创建子图，按行排列
    graphs = []
    for g in g_list:
        G = nx.Graph()
        G.add_edges_from(g["graph"])
        graphs.append(G)
    titles = ['Rank by ' + model]
    titles.extend(nged_list)
    subtitles = ['', 1,2,3,4,5,6,41,600]
    # 遍历每个图并绘制在相应的子图中
    for i, g in enumerate(graphs):
        ax = axs[i]  # 选择对应的子图
        # 如果没有指定位置，则使用 spring 布局
        pos = nx.spring_layout(g, seed=42)  # 固定随机种子的布局
        # 绘制节点和边
        nx.draw(g, pos, with_labels=False, 
                ax=ax, node_size=500, edge_color='black')
        ax.set_title(titles[i], fontsize=18)
        ax.text(0.5, -0.1, subtitles[i], fontsize=16, ha='center', transform=ax.transAxes)
    plt.savefig("./pictures/rank/Linux/rank_{}_{}_{}.png".format(model, dataset, gid_list[0]))
    
def cal_pk(num, pre, gt):
    """
    分别按真实值和预测值排序，并取前k个，统计前k个中有几个重合
    因为不考虑精确匹配，只考虑前k个是否重合，所以真实值和预测值的排序无论先后
    """
    tmp = list(zip(gt, pre))
    tmp.sort()  # 按真实值从小到大排
    beta = []
    for i, p in enumerate(tmp):  # 标记
        beta.append((p[1], p[0], i))
    beta.sort()  # 按预测值从小到大排
    ans = 0
    for i in range(num):  # 根据标记，统计gt的前n中有几个pre的前n
        if beta[i][2] < num:
            ans += 1
    return ans / num

def draw_rank(model, dataset):
    """
    [(id1,[gid1, gid2,...],[ged1, ged2, ...],[pre1, pre2,...],[nged1,nged2,...]),  420个图对
     (id2,[gid1, gid2,...],[ged1, ged2, ...],[pre1, pre2,...],[nged1,nged2,...]),  420个图对
     ...
     (id140,[gid1, gid2,...],[ged1, ged2, ...],[pre1, pre2,...]),],[nged1,nged2,...]
    """
    data_list = load_rank_data(model, dataset)
    # """找p10最大的测试组"""
    # p10_max = 0  # 当前最大值
    # p10_list = []  # 保存所有p10结果
    # i_list = []  # 保存最大值的索引（存在多个）
    # for i, data in enumerate(data_list):  # 找到pk值最大的测试组
    #     p10 = cal_pk(6, data[2], data[3])  # 计算pk值
    #     p10_list.append(p10)  # 保存，后续计算均值
    #     if p10 > p10_max:
    #         p10_max = p10
    #         i_list = []
    #         i_list.append(i)
    #     elif p10 == p10_max:  # 如果相同，一并保存
    #         i_list.append(i)
    # print(i_list, p10_max, round(np.mean(p10_list), 3))
    # """在p10最大的测试组里找平均ged最小的测试组"""
    # x_list = []
    # for j in i_list:
    #     x = round(np.mean(data_list[j][2][:6]), 3)  # 计算平均ged
    #     x_list.append((x, j, data_list[j][0]))
    # x_list.sort()  # 按平均ged从升序排序
    # print(x_list)
    # print(len(x_list))
    """绘图"""
    # [118, 69, 140, 193, 157]
    # [118, 69, 140, 193, 157, 56, 92, 164, 182]
    for x in [80]:   
        # print(x)
        j = x  # AIDS:28/80/30/5 从x_list[i][1]中选择一个
        gid_list = [data_list[j][0]]
        nged_list = []
        if dataset == "AIDS":
            gid_list.extend([data_list[j][1][x] for x in [0,1,2,3,4,5,20,419]])
            nged_list.extend([data_list[j][4][x] for x in [0,1,2,3,4,5,20,419]])
        else: 
            gid_list.extend([data_list[j][1][x] for x in [0,1,2,3,4,5,40,599]])
            nged_list.extend([data_list[j][4][x] for x in [0,1,2,3,4,5,40,599]])
        print("需要绘制的图ID列表:\n", gid_list)
        g_list = []
        for k, gid in enumerate(gid_list):
            if k == 0:
                g = json.load(open('./json_data/{}/test/{}.json'.format(dataset, gid), 'r'))
            else: g = json.load(open('./json_data/{}/train/{}.json'.format(dataset, gid), 'r'))
            g_list.append(g)
        if dataset == "AIDS":
            draw_aids_rank(g_list, gid_list, nged_list, model, dataset)
        else:
            draw_linux_rank(g_list, gid_list, nged_list, model, dataset)

def draw_ablation_bar():
    title = ["No-gs", "No-cost"]
    # 数据
    datasets = ['AIDS', 'Linux']
    values_mygnn = [0.637, 0.208]
    values_no_gs = [0.655, 0.305]
    # values_no_pruning = [1.491, 0.229]

    # 设置柱状图的宽度和x轴位置
    bar_width = 0.2
    x = np.arange(len(datasets))
    # gap = 0.3  # 每组之间的间隙
    # x = np.arange(len(datasets)) * (bar_width * 3 + gap)

    # 绘制并列柱状图
    fig, ax = plt.subplots()

    # 绘制每组柱状图
    bars1 = ax.bar(x - bar_width, values_mygnn, width=bar_width, label='MyGNN', color='salmon')
    bars2 = ax.bar(x, values_no_gs, width=bar_width, label='No Gumbel-sinkhorn', color='lightblue')
    # bars3 = ax.bar(x + bar_width, values_no_pruning, width=bar_width, label='No GED Lower Bound Pruning', color='lightgreen')

    # 设置标题和标签
    ax.set_xlabel('Dataset')
    ax.set_ylabel('GED MAE')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.0)
    ax.set_title('GED MAE Comparison')

    # 添加每个柱的数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2 , yval + 0.05, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

    # 显示图例
    ax.legend()
    plt.savefig("./pictures/ablation/ablation_{}.png".format(1))
    

def f():
    pass

if __name__ == "__main__":
    # 在“GEDGNN/experiments/Overall Performance”目录下运行
    # load_data2()
    # draw_rank('MyGNN3', 'AIDS')
    
    # draw_ablation_bar()
    pass

    