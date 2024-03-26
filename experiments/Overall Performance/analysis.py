import matplotlib.pyplot as plt
import numpy as np

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

def MAE_plot(data_list1, data_list2):
    # 准备数据
    mygnn_mae = []
    gedgnn_mae = []
    mygnn_arg = 0
    gedgnn_arg = 0
    x = np.arange(1, 21) 
    for item in data_list1:
        mygnn_mae.append(float(item[6]))
        mygnn_arg += float(item[6])
    for item in data_list2:
        gedgnn_mae.append(float(item[6]))
        gedgnn_arg += float(item[6])
    mygnn_arg /= 20
    gedgnn_arg /= 20
    
    # 设置画布大小
    plt.figure(figsize=(20, 10), dpi=100)
    
    # 坐标刻度
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0.2, 2, 0.1))
    # plt.ylim((0.5, 1.5))  坐标限制
    
    # 坐标描述
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('MAE', fontsize=20)
    
    # 设置数据标签位置及大小
    for a, b in zip(x, mygnn_mae):
        plt.text(a, b+0.005, b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(x, gedgnn_mae):
        plt.text(a, b+0.005, b, ha='center', va='bottom', fontsize=10) 
    
    # 画图并保存
    plt.plot(x, mygnn_mae, marker='o', markersize=5, label='mygnn_mae')
    plt.plot(x, gedgnn_mae, marker='o', markersize=5, label='gedgnn_mae')
    plt.hlines(y=mygnn_arg, xmin=0, xmax=20, edgecolors='r', label='mygnn_arg=%.3f'%mygnn_arg)
    plt.hlines(y=gedgnn_arg, xmin=0, xmax=20, label='gedgnn_arg=%.3f'%gedgnn_arg)
    
    plt.legend()
    plt.savefig('/home/zhj/code/z/pic/MAE.png')
    
def Accuracy_plot(data_list1, data_list2):
    # 准备数据
    mygnn_acc = []
    gedgnn_acc = []
    mygnn_arg = 0
    gedgnn_arg = 0
    x = np.arange(1, 21) 
    for item in data_list1:
        mygnn_acc.append(float(item[7]))
        mygnn_arg += float(item[7])
    for item in data_list2:
        gedgnn_acc.append(float(item[7]))
        gedgnn_arg += float(item[7])
    mygnn_arg /= 20
    gedgnn_arg /= 20
    
    # 设置画布大小
    plt.figure(figsize=(20, 10), dpi=100)
    
    # 坐标刻度
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0.2, 0.5, 0.03))
    # plt.ylim((0.5, 1.5))  坐标限制
    
    # 坐标描述
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    
    # 设置数据标签位置及大小
    for a, b in zip(x, mygnn_acc):
        plt.text(a, b+0.002, b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(x, gedgnn_acc):
        plt.text(a, b+0.002, b, ha='center', va='bottom', fontsize=10) 
    
    # 画图并保存
    plt.plot(x, mygnn_acc, marker='o', markersize=5, label='mygnn_acc')
    plt.plot(x, gedgnn_acc, marker='o', markersize=5, label='gedgnn_acc')
    plt.hlines(y=mygnn_arg, xmin=0, xmax=20, edgecolors='r', label='mygnn_arg=%.3f'%mygnn_arg)
    plt.hlines(y=gedgnn_arg, xmin=0, xmax=20, label='gedgnn_arg=%.3f'%gedgnn_arg)
    
    plt.legend()
    plt.savefig('/home/zhj/code/z/pic/Accuracy.png')

def p_plot(data_list1, data_list2):
    # 准备数据
    mygnn_p10 = []
    gedgnn_p10 = []
    mygnn_arg = 0
    gedgnn_arg = 0
    x = np.arange(1, 21) 
    for item in data_list1:
        mygnn_p10.append(float(item[11]))
        mygnn_arg += float(item[11])
    for item in data_list2:
        gedgnn_p10.append(float(item[11]))
        gedgnn_arg += float(item[11])
    mygnn_arg /= 20
    gedgnn_arg /= 20
    
    # 设置画布大小
    plt.figure(figsize=(20, 10), dpi=100)
    
    # 坐标刻度
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0.5, 1, 0.05))
    # plt.ylim((0.5, 1.5))  坐标限制
    
    # 坐标描述
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('p@10/20', fontsize=20)
    
    # 设置数据标签位置及大小
    for a, b in zip(x, mygnn_p10):
        plt.text(a, b+0.002, b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(x, gedgnn_p10):
        plt.text(a, b+0.002, b, ha='center', va='bottom', fontsize=10) 
    
    # 画图并保存
    plt.plot(x, mygnn_p10, marker='o', markersize=5, label='mygnn_p10')
    plt.plot(x, gedgnn_p10, marker='o', markersize=5, label='gedgnn_p10')
    plt.hlines(y=mygnn_arg, xmin=0, xmax=20, edgecolors='r', label='mygnn_arg=%.3f'%mygnn_arg)
    plt.hlines(y=gedgnn_arg, xmin=0, xmax=20, label='gedgnn_arg=%.3f'%gedgnn_arg)
    
    plt.legend()
    plt.savefig('/home/zhj/code/z/pic/p@10.png')

if __name__ == "__main__":
    path1 = '/home/zhj/code/z/mygnn.txt'
    path2 = '/home/zhj/code/z/gedgnn.txt'
    data_list1 = load_data(path1)
    data_list2 = load_data(path2)
    # MAE_plot(data_list1, data_list2)
    # Accuracy_plot(data_list1, data_list2)
    p_plot(data_list1, data_list2)
    






