import math
import dgl
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCNConv, GINConv
from torch_geometric.nn.glob import global_add_pool
from layers import AttentionModule, TensorNetworkModule, sinkhorn, MatchingModule, GraphAggregationLayer, Mlp, my_gumbel_softmax, gumbel_sinkhorn
from math import exp
from GedMatrix import GedMatrixModule, SimpleMatrixModule
from lap import lapjv


class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)

        # bias
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)

        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)
        # self.bias_model = torch.nn.Linear(2, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        # features = torch.sigmoid(features)
        return features

    def ntn_pass(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)
        return scores

    def forward(self, data, return_ged=False):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :param is_testing: pass
        :param predict_value: pass
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        scores = self.ntn_pass(abstract_features_1, abstract_features_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = F.relu(self.fully_connected_first(scores))
        scores = F.relu(self.fully_connected_second(scores))
        scores = F.relu(self.fully_connected_third(scores))
        score = torch.sigmoid(self.scoring_layer(scores).view(-1))

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item()


class GPN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GPN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.args.gnn_operator = 'gin'

        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.matching_1 = MatchingModule(self.args)
        self.matching_2 = MatchingModule(self.args)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        # using_dropout = self.training
        using_dropout = False
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=using_dropout)
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=using_dropout)
        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        tmp_feature_1 = abstract_features_1
        tmp_feature_2 = abstract_features_2

        abstract_features_1 = torch.sub(tmp_feature_1, self.matching_2(tmp_feature_2))
        abstract_features_2 = torch.sub(tmp_feature_2, self.matching_1(tmp_feature_1))

        abstract_features_1 = torch.abs(abstract_features_1)
        abstract_features_2 = torch.abs(abstract_features_2)

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item()


class GedGNN(torch.nn.Module):

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GedGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.args.gnn_operator = 'gin'

        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1, track_running_stats=False))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2, track_running_stats=False))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.mapMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)
        self.costMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)
        # self.costMatrix = SimpleMatrixModule(self.args.filters_3)

        # bias
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)

        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)
        # self.bias_model = torch.nn.Linear(2, 1)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        # features = torch.sigmoid(features)
        return features

    def get_bias_value(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))
        score = self.scoring_layer(scores).view(-1)
        return score

    @staticmethod
    def ged_from_mapping(matrix, A1, A2, f1, f2):
        # edge loss
        A_loss = torch.mm(torch.mm(matrix.t(), A1), matrix) - A2
        # label loss
        F_loss = torch.mm(matrix.t(), f1) - f2
        mapping_ged = ((A_loss * A_loss).sum() + (F_loss * F_loss).sum()) / 2.0
        return mapping_ged.view(-1)

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :param is_testing: whether return ged value together with ged score
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)
        map_matrix = self.mapMatrix(abstract_features_1, abstract_features_2)
        
        # calculate ged using map_matrix
        m = torch.nn.Softmax(dim=1)
        soft_matrix = m(map_matrix) * cost_matrix
        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2)
        score = torch.sigmoid(soft_matrix.sum() + bias_value)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item(), map_matrix


class MyGNN3(torch.nn.Module):

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(MyGNN3, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.args.gnn_operator = 'gin'

        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.args.init_features, self.args.filters_1),  # self.number_labels
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1, track_running_stats=False))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2, track_running_stats=False))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
            self.compress = torch.nn.Linear(224, self.args.final_features)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.costMatrix = GedMatrixModule(self.args.final_features, self.args.hidden_dim)  # self.args.filters_3  cat

        # bias
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)
        
        # LRL模块
        self.transform1 = torch.nn.Linear(self.args.final_features, self.args.lrl_hiddim)  # self.args.filters_3  cat
        self.relu1 = torch.nn.ReLU()
        self.transform2 = torch.nn.Linear(self.args.lrl_hiddim, self.args.lrl_hiddim)
        
        # one-hot初始化
        self.embedding = torch.nn.Linear(self.number_labels, self.args.init_features)
    
    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        # features = torch.sigmoid(features)
        return features
    
    def convolutional_pass_2(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features_1 = self.convolution_1(features, edge_index)
        features_1 = torch.nn.functional.relu(features_1)
        features_1 = torch.nn.functional.dropout(features_1,
                                               p=self.args.dropout,
                                               training=self.training)  # [n, 128]
        cat_features = features_1
        features_2 = self.convolution_2(features_1, edge_index)
        features_2 = torch.nn.functional.relu(features_2)
        features_2 = torch.nn.functional.dropout(features_2,
                                               p=self.args.dropout,
                                               training=self.training)  # [n, 64]
        cat_features = torch.cat((cat_features, features_2), dim = 1)
        features_3 = self.convolution_3(features_2, edge_index)  # [n, 32]
        features_3 = torch.nn.functional.relu(features_3)
        features_3 = torch.nn.functional.dropout(features_3,
                                               p=self.args.dropout,
                                               training=self.training)  # [n, 64]
        cat_features = torch.cat((cat_features, features_3), dim = 1)  # [n, 224]
        cat_features = self.compress(cat_features)  # [n, 32]
        return cat_features

    def get_bias_value(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)  # 32*1
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)  # 1*16
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))  # 16-16
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))  # 16-8
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))  # 8-4
        score = self.scoring_layer(scores).view(-1)  # 4-1
        return score

    @staticmethod
    def ged_from_mapping(matrix, A1, A2, f1, f2):
        # edge loss
        A_loss = torch.mm(torch.mm(matrix.t(), A1), matrix) - A2
        # label loss
        F_loss = torch.mm(matrix.t(), f1) - f2
        mapping_ged = ((A_loss * A_loss).sum() + (F_loss * F_loss).sum()) / 2.0
        return mapping_ged.view(-1)
    
    def LRL(self, abstract_features_1, abstract_features_2):
        """经过LRL和gumbel-sinkhorn得到置换矩阵
        Args:
            abstract_features_1 (_type_): 图1节点嵌入 [n1, 32]
            abstract_features_2 (_type_): 图2节点嵌入 [n2, 32]
        Returns:
            _type_: 相似度矩阵
        """
        n1 = abstract_features_1.shape[0]
        n2 = abstract_features_2.shape[0]
        max_size = max(n1, n2)
        # LRL
        emb_1 = self.transform2(self.relu1(self.transform1(abstract_features_1)))  # [n1, 64]
        emb_2 = self.transform2(self.relu1(self.transform1(abstract_features_2)))  # [n2, 64]
        sinkhorn_input = torch.matmul(emb_1, emb_2.permute(1,0))  # [n1, n2]
        # 填充
        sinkhorn_input = F.pad(sinkhorn_input, pad=(0, max_size-n2, 0, max_size-n1))  # [max_size, max_size], 左右上下
        # 虽然在上面mask掉了填充部分，但经过gumbel后，mask的部分仍会有值，需要A_match方面进行mask
        return sinkhorn_input
    
    def Cross_(self, abstract_features_1, abstract_features_2):
        """通过嵌入计算相似度矩阵
        Args:
            abstract_features_1 (_type_): 图1嵌入 [n1, 32]
            abstract_features_2 (_type_): 图2嵌入 [n1, 32]
            flag (_type_): 节点嵌入 or 边嵌入
        Returns:
            _type_: 相似度矩阵(通过mask消除填充向量的影响, 同时因为之后会和置换矩阵相乘, 置换矩阵不用再mask)
            取值范围(-无穷，+无穷)
        """
        n1 = abstract_features_1.shape[0]
        n2 = abstract_features_2.shape[0]
        max_size = max(n1, n2)
        # 填充
        abstract_features_1 = F.pad(abstract_features_1, pad=(0,0,0,max_size-n1))  # [max_size, 32]
        abstract_features_2 = F.pad(abstract_features_2, pad=(0,0,0,max_size-n2))  # [max_size, 32]
        # 交互
        m = self.costMatrix(abstract_features_1, abstract_features_2)  # [max_size, max_size]
        if n1 < n2:
            # mask
            mask = torch.cat((torch.ones(n2).repeat(n1,1), torch.zeros(n2).repeat(n2-n1,1))).to(self.device)  # [max_size, max_size]
            cost_matrix = torch.mul(mask, m)
            # deletion cost
            pooled_features_2 = torch.mean(abstract_features_2, dim=0, keepdim=True).transpose(0,1)  # d*1
            del_cost_list = torch.mm(abstract_features_2, pooled_features_2).transpose(0,1)  # [n2,d]*[d,1]=[n2,1] [1,n2]
            del_cost_matrix = torch.cat((torch.zeros(n2).repeat(n1,1), del_cost_list.repeat(n2-n1,1)))  # n2*n2
            return cost_matrix + del_cost_matrix
        elif n1 == n2: 
            return m
        else: 
            print("\nerror\n")

    def Cross(self, abstract_features_1, abstract_features_2):
        n1 = abstract_features_1.shape[0]
        n2 = abstract_features_2.shape[0]
        if n1 < n2:  # 图1节点数小于等于图2
            # 计算增删代价
            pooled_features_2 = torch.mean(abstract_features_2, dim=0, keepdim=True)  # 1*d
            abstract_features_1 = torch.cat((abstract_features_1, pooled_features_2.repeat(n2-n1,1)))  # n2*d
        elif n1 > n2: print("\nerror\n") 
        # 交互
        m = self.costMatrix(abstract_features_1, abstract_features_2)  # [max_size, max_size]
        return m


    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :param is_testing: whether return ged value together with ged score
        :return score: Similarity score.
        """
        self.device = data["device"]
        edge_index_1 = data["edge_index_1"]  # (torch.Tensor)2*n
        edge_index_2 = data["edge_index_2"]  # (torch.Tensor)2*m
        features_1 = data["features_1"]  # (torch.Tensor)n*29
        features_2 = data["features_2"]  # (torch.Tensor)m*29
        features_1 = self.embedding(features_1)
        features_2 = self.embedding(features_2)
        # 计算节点嵌入
        abstract_features_1 = self.convolutional_pass_2(edge_index_1, features_1)  # [n1, 224]
        abstract_features_2 = self.convolutional_pass_2(edge_index_2, features_2)  # [n2, 224]
        
        # 计算节点成本矩阵
        cost_matrix = self.Cross(abstract_features_1, abstract_features_2)  # max*max mask (max-n)*(max-m)
        
        # 计算节点对齐矩阵
        LRL_map_matrix = self.LRL(abstract_features_1, abstract_features_2)  # max*max
        node_alignment = gumbel_sinkhorn(LRL_map_matrix, tau=0.1)  # # max*max
     
        # 计算map_matrix和bias
        # m = torch.nn.Softmax(dim=1)
        soft_matrix = node_alignment * cost_matrix
        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2)
        score = torch.sigmoid(soft_matrix.sum() + bias_value)  # 0~1
        
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False

        return score, pre_ged.item(), (LRL_map_matrix.detach().cpu().numpy(), node_alignment.detach().cpu().numpy())


class TaGSim(torch.nn.Module):
    """
    TaGSim: Type-aware Graph Similarity Learning and Computation
    https://github.com/jiyangbai/TaGSim
    """
    def __init__(self, args, number_of_labels):
        super(TaGSim, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        self.gal1 = GraphAggregationLayer()
        self.gal2 = GraphAggregationLayer()
        self.feature_count = self.args.tensor_neurons

        self.tensor_network_nc = TensorNetworkModule(self.args, 2 * self.number_labels)
        self.tensor_network_in = TensorNetworkModule(self.args, 2 * self.number_labels)
        self.tensor_network_ie = TensorNetworkModule(self.args, 2 * self.number_labels)

        self.fully_connected_first_nc = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_nc = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_nc = torch.nn.Linear(8, 4)
        self.scoring_layer_nc = torch.nn.Linear(4, 1)

        self.fully_connected_first_in = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_in = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_in = torch.nn.Linear(8, 4)
        self.scoring_layer_in = torch.nn.Linear(4, 1)

        self.fully_connected_first_ie = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_ie = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_ie = torch.nn.Linear(8, 4)
        self.scoring_layer_ie = torch.nn.Linear(4, 1)

    def gal_pass(self, edge_index, features):
        hidden1 = self.gal1(features, edge_index)
        hidden2 = self.gal2(hidden1, edge_index)

        return hidden1, hidden2

    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        n1, n2 = data["n1"], data["n2"]

        adj_1 = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.shape[1]), (n1, n1)).to_dense()
        adj_2 = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.shape[1]), (n2, n2)).to_dense()
        # remove self-loops
        adj_1 = adj_1 * (1.0 - torch.eye(n1))
        adj_2 = adj_2 * (1.0 - torch.eye(n2))

        graph1_hidden1, graph1_hidden2 = self.gal_pass(adj_1, features_1)
        graph2_hidden1, graph2_hidden2 = self.gal_pass(adj_2, features_2)

        graph1_01concat = torch.cat([features_1, graph1_hidden1], dim=1)
        graph2_01concat = torch.cat([features_2, graph2_hidden1], dim=1)
        graph1_12concat = torch.cat([graph1_hidden1, graph1_hidden2], dim=1)
        graph2_12concat = torch.cat([graph2_hidden1, graph2_hidden2], dim=1)

        graph1_01pooled = torch.sum(graph1_01concat, dim=0).unsqueeze(1)
        graph1_12pooled = torch.sum(graph1_12concat, dim=0).unsqueeze(1)
        graph2_01pooled = torch.sum(graph2_01concat, dim=0).unsqueeze(1)
        graph2_12pooled = torch.sum(graph2_12concat, dim=0).unsqueeze(1)

        scores_nc = self.tensor_network_nc(graph1_01pooled, graph2_01pooled)
        scores_nc = torch.t(scores_nc)

        scores_nc = torch.nn.functional.relu(self.fully_connected_first_nc(scores_nc))
        scores_nc = torch.nn.functional.relu(self.fully_connected_second_nc(scores_nc))
        scores_nc = torch.nn.functional.relu(self.fully_connected_third_nc(scores_nc))
        score_nc = torch.sigmoid(self.scoring_layer_nc(scores_nc))

        scores_in = self.tensor_network_in(graph1_01pooled, graph2_01pooled)
        scores_in = torch.t(scores_in)

        scores_in = torch.nn.functional.relu(self.fully_connected_first_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_second_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_third_in(scores_in))
        score_in = torch.sigmoid(self.scoring_layer_in(scores_in))

        scores_ie = self.tensor_network_ie(graph1_12pooled, graph2_12pooled)
        scores_ie = torch.t(scores_ie)

        scores_ie = torch.nn.functional.relu(self.fully_connected_first_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_second_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_third_ie(scores_ie))
        score_ie = torch.sigmoid(self.scoring_layer_ie(scores_ie))

        score = torch.cat([score_nc.view(-1), score_in.view(-1), score_ie.view(-1)])
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.sum().item()


class GOTSim(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GOTSim, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
    
    def setup_layers(self):
        """
        Creating the layers.
        """ 
        self.gcn_layers = torch.nn.ModuleList([])
        self.args.gcn_size = [128, 64, 32]
        num_ftrs = self.number_labels
        self.num_gcn_layers = len(self.args.gcn_size)
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i]))
            num_ftrs = self.args.gcn_size[i]
        
        self.ot_scoring_layer = torch.nn.Linear(self.num_gcn_layers, 1)
        
        # Params for insertion and deletion embeddings 
        self.insertion_params, self.deletion_params = torch.nn.ParameterList([]), torch.nn.ParameterList([])
        for i in range(self.num_gcn_layers):
            self.insertion_params.append(torch.nn.Parameter(torch.ones(self.args.gcn_size[i])))
            self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.args.gcn_size[i])))
        
    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param graph: DGL graph.
        :param features: Feature matrix.
        :return features: List of abstract feature matrices.
        """
        abstract_feature_matrices = []
        for i in range(self.num_gcn_layers-1):
            features = self.gcn_layers[i](features, edge_index)
            abstract_feature_matrices.append(features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)
            
        features = self.gcn_layers[-1](features, edge_index)
        abstract_feature_matrices.append(features)
        return abstract_feature_matrices
    
    def dense_wasserstein_distance(self, cost_matrix):
        num_pts = len(cost_matrix)  # n1+n2
        C_cpu = cost_matrix.detach().cpu().numpy()  # 移动到 CPU，并转换为 NumPy 数组，以便使用后续的 LAPJV 算法
        lowest_cost, col_ind_lapjv, row_ind_lapjv = lapjv(C_cpu)  # 调用 lapjv 函数解决线性分配问题（最优匹配）,每个行元素的最佳匹配列索引
        
        loss = torch.tensor([0.0]).to(self.device)
        for i in range(num_pts):
            loss += cost_matrix[i,col_ind_lapjv[i]]
                    
        return loss/num_pts
    
    def forward(self, data):
        
        self.device = data["device"]
        edge_index_1 = data["edge_index_1"]  # (torch.Tensor)2*n
        edge_index_2 = data["edge_index_2"]  # (torch.Tensor)2*m
        features_1 = data["features_1"]  # (torch.Tensor)n*29
        features_2 = data["features_2"]  # (torch.Tensor)m*29
        n1 = data["n1"]
        n2 = data["n2"]

        abstract_features_list_1 = self.convolutional_pass(edge_index_1, features_1)  # [n1*128, n1*64, n1*32]
        abstract_features_list_2 = self.convolutional_pass(edge_index_2, features_2)  # [n2*128, n2*64, n2*32]
        
        main_similarity_matrices_list = [-torch.mm(abstract_features_list_1[i], 
                                                          abstract_features_list_2[i].transpose(0,1)) 
                                                 for i in range(self.num_gcn_layers)]  # n1*n2

        # these are matrix with 0 on the diagonal and inf cost on off-diagonal
        insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) - torch.diag(torch.ones(n1))).to(self.device)  # n1*n1
        deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) - torch.diag(torch.ones(n2))).to(self.device)  # n2*n2

        # 计算删除代价
        deletion_similarity_matrices_list = [
            torch.diag(-torch.matmul(abstract_features_list_1[i], self.deletion_params[i]))  # [n1,128] * [128,1]
                + insertion_constant_matrix
            for i in range(self.num_gcn_layers)
        ]  # n1*n1

        # 计算新增代价
        insertion_similarity_matrices_list = [
            torch.diag(-torch.matmul(abstract_features_list_2[i], self.insertion_params[i])) 
                + deletion_constant_matrix
            for i in range(self.num_gcn_layers)
        ]  # n2*n2
        
        dummy_similarity_matrices_list = [
            torch.zeros(n2, n1, dtype=abstract_features_list_1[i].dtype).to(self.device)
            for i in range(self.num_gcn_layers)
        ]  # n2*n1
        
        # 四个矩阵拼接成一个大矩阵[n1+n2, n1+n2]
        similarity_matrices_list = [
            torch.cat(
                (
                    torch.cat((main_similarity_m, deletion_similarity_m), dim=1),
                    torch.cat((insertion_similarity_m, dummy_similarity_m), dim=1)
                ), dim=0)
            for main_similarity_m, deletion_similarity_m, insertion_similarity_m, dummy_similarity_m 
            in zip(main_similarity_matrices_list, deletion_similarity_matrices_list, 
                   insertion_similarity_matrices_list, dummy_similarity_matrices_list)
        ]  # [n1+n2, n1+n2]
        
        # 通过线性分配函数得到最小代价
        matching = [self.dense_wasserstein_distance(s) 
                         for s in similarity_matrices_list]  # num_gcn_layers*1
        
        matching_cost = matching
        
        matching_cost = 2 * torch.cat(matching_cost) / (n1 + n2)        
        score_logits = self.ot_scoring_layer(matching_cost)
        score = torch.sigmoid(score_logits)
        
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False

        return score, pre_ged.item(), matching_cost


class Readout(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(Readout, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
    
    def setup_layers(self):
        """
        Creating the layers.
        """

        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)
    
    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        # features = torch.sigmoid(features)
        return features
    
    def ntn_pass(self, pooled_features_1, pooled_features_2):
        scores = self.tensor_network(pooled_features_1, pooled_features_2)  # 1*16
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))  # 16-16
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))  # 16-8
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))  # 8-4
        score = self.scoring_layer(scores).view(-1)  # 4-1
        return score
    
    def forward(self, data):
        self.device = data["device"]
        edge_index_1 = data["edge_index_1"]  # (torch.Tensor)2*n
        edge_index_2 = data["edge_index_2"]  # (torch.Tensor)2*m
        features_1 = data["features_1"]  # (torch.Tensor)n*29
        features_2 = data["features_2"]  # (torch.Tensor)m*29
        
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)  # n1*32
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)  # n2*32

        if self.args.readout == "max":
            pooled_features_1 = torch.max(abstract_features_1, dim=0, keepdim=True)[0].transpose(0,1)  # 32*1
            pooled_features_2 = torch.max(abstract_features_2, dim=0, keepdim=True)[0].transpose(0,1)
        elif self.args.readout == "mean":
            pooled_features_1 = torch.mean(abstract_features_1, dim=0, keepdim=True).transpose(0,1)
            pooled_features_2 = torch.mean(abstract_features_2, dim=0, keepdim=True).transpose(0,1)
            
        score_logit = self.ntn_pass(pooled_features_1, pooled_features_2)
        score = torch.sigmoid(score_logit)
        
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False

        return score, pre_ged.item(), score_logit
        # return score.view(-1), score_logit.view(-1)