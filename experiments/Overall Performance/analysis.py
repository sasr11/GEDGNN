import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(path):
    x = "mse"
    data_list = []
    with open(path, 'r') as file: 
        lines = file.readlines() 
        i = 0
        for i in range(len(lines)): 
            if x in lines[i]:
                s = lines[i+1]
                data_list.append(s.split())
            i += 1
    return data_list
    
def MAE(data_list, label_list):
    # 取出数据中的MAE的值
    show_data = []
    show_arg = []
    for exp in data_list: # 每次实验结果
        data = []
        arg = 0
        for item in exp:  # 实验中每个epoch的结果
            a = float(item[6])  # 取出MAE的值
            data.append(a)
            arg += a
        show_data.append(data)
        show_arg.append(arg/20)
        
    # 设置画布大小
    plt.figure(figsize=(20, 10), dpi=100)
    
    # 坐标刻度
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0.2, 2, 0.1))
    
     # 坐标描述
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('MAE', fontsize=20)
    
    # 设置数据标签位置及大小
    x = np.arange(1, 21)
    for y in show_data:
        for a, b in zip(x, y):
            plt.text(a, b+0.002, b, ha='center', va='bottom', fontsize=10)
    # 画图并保存
    for d, a, l in zip(show_data, show_arg, label_list):
        plt.plot(x, d, marker='o', markersize=5, label=l+'_mae')
        plt.hlines(y=a, xmin=0, xmax=20, label=l+'_arg=%.3f'%a)
        
    plt.legend()
    plt.savefig('experiments/Overall Performance/result/pic/MAE.png')

def Accuracy(data_list, label_list):
    # 取出数据中的Accuracy的值
    show_data = []
    show_arg = []
    for exp in data_list: # 每次实验结果
        data = []
        arg = 0
        for item in exp:  # 实验中每个epoch的结果
            a = float(item[7])  # 取出Accuracy的值
            data.append(a)
            arg += a
        show_data.append(data)
        show_arg.append(arg/20)
        
    # 设置画布大小
    plt.figure(figsize=(20, 10), dpi=100)
    
    # 坐标刻度
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0.2, 0.5, 0.03))
    
     # 坐标描述
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    
    # 设置数据标签位置及大小
    x = np.arange(1, 21)
    for y in show_data:
        for a, b in zip(x, y):
            plt.text(a, b+0.002, b, ha='center', va='bottom', fontsize=10)
    # 画图并保存
    for d, a, l in zip(show_data, show_arg, label_list):
        plt.plot(x, d, marker='o', markersize=5, label=l+'_acc')
        plt.hlines(y=a, xmin=0, xmax=20, label=l+'_arg=%.3f'%a)
        
    plt.legend()
    plt.savefig('experiments/Overall Performance/result/pic/Accuracy.png')

def p10(data_list, label_list):
    """处理数据, 生成p10的统计图
    Args:
        data_list (_type_): 数据列表，每一个元素代表一次实验结果
        label_list (_type_): 每次实验结果的标签
    """
    # 取出数据中的p10的值
    show_data = []
    show_arg = []
    for exp in data_list: # 每次实验结果
        data = []
        arg = 0
        for item in exp:  # 实验中每个epoch的结果
            a = float(item[11])  # 取出p10的值
            data.append(a)
            arg += a
        show_data.append(data)
        show_arg.append(arg/20)
        
    # 设置画布大小
    plt.figure(figsize=(20, 10), dpi=100)
    
    # 坐标刻度
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0.5, 1, 0.05))
    
     # 坐标描述
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('p@10', fontsize=20)
    
    # 设置数据标签位置及大小
    x = np.arange(1, 21)
    for y in show_data:
        for a, b in zip(x, y):
            plt.text(a, b+0.002, b, ha='center', va='bottom', fontsize=10)
    # 画图并保存
    for d, a, l in zip(show_data, show_arg, label_list):
        plt.plot(x, d, marker='o', markersize=5, label=l+'_p10')
        plt.hlines(y=a, xmin=0, xmax=20, label=l+'_arg=%.3f'%a)
        
    plt.legend()
    plt.savefig('experiments/Overall Performance/result/pic/p@10.png')

if __name__ == "__main__":
    # print(os.getcwd())
    path_list = ['experiments/Overall Performance/result/result_GedGNN_AIDS_2404130954.txt',
                 'experiments/Overall Performance/result/result_MyGNN2_AIDS_2404121719.txt',]
    label_list = ['gedgnn_no_bce', 'mygnn2']
    data_list = []
    for path in path_list:
        data_list.append(load_data(path))
    MAE(data_list, label_list)
    Accuracy(data_list, label_list)
    p10(data_list, label_list)
    






