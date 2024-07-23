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


class MyGNN(torch.nn.Module):

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(MyGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.args.gnn_operator = 'gin'
        if self.args.gnn_operator == 'gin':
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
        
        # My
        self.mlp = Mlp(self.args.filters_3 * 2)

    def convolutional_pass(self, edge_index, features):
        """
        图卷积
        :param edge_index: 边信息. 2*m
        :param features: 特征矩阵 n*29
        :return features: Abstract feature matrix. n*32
        """
        # n*29 -> n*128
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)
        # n*128 -> n*64
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)
        # n*64 -> n*32
        features = self.convolution_3(features, edge_index)
        # features = torch.sigmoid(features)
        return features

    def graph_embedding(self, abstract_features):
        """
        图卷积
        :param abstract_features_1: 节点embedding n*filters_3
        :return pooled_features: 图级embedding filters_3*1  ??
        """
        pooled_features = self.attention(abstract_features)
        return pooled_features

    def get_bias_value(self, pooled_features_1, pooled_features_2):
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))
        score = self.scoring_layer(scores).view(-1)
        return score

    def get_masked_index(self, n_s, n_l, abstract_features, pooled_features):
        """
        计算节点保留概率
        :param n_s: 较小图的节点数量
        :param abstract_features: 节点向量 n*filters_3
        :param pooled_features: 图级向量 filters_3*1
        :return probability: 节点数较多的图的节点保留概率 1*n
        """
        # 扩展图级向量到与节点数量相同的大小
        pooled_features = pooled_features.squeeze()  # 1*filters_3
        expanded_pooled_features = pooled_features.repeat(n_l, 1)  # n*filters_3
        # 使用图级特征与每个节点特征进行拼接
        concatenated_features = torch.cat((abstract_features, expanded_pooled_features), dim=1)  # n*[filters_3*2]
        # 使用MLP模型计算保留概率
        outputs = self.mlp(concatenated_features).squeeze()  # n*1 
        # 使用gumbel_softmax进行软采样
        masked_index = my_gumbel_softmax(outputs, n_s)
        return masked_index
        
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
        n1 = data["n1"]
        n2 = data["n2"]
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        # node embeddings
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        # graph embeddings
        pooled_features_1 = self.graph_embedding(abstract_features_1)
        pooled_features_2 = self.graph_embedding(abstract_features_2)
        # 交互
        cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)
        map_matrix = self.mapMatrix(abstract_features_1, abstract_features_2)
        
        # calculate masked_index
        if n1 > n2:
            masked_index = self.get_masked_index(n2, n1, abstract_features_1, pooled_features_2)
        elif n1 < n2:
            masked_index = self.get_masked_index(n1, n2, abstract_features_2, pooled_features_1)
        else:
            masked_index = None
        
        # calculate ged using map_matrix
        m = torch.nn.Softmax(dim=1)
        soft_matrix = m(map_matrix) * cost_matrix
        bias_value = self.get_bias_value(pooled_features_1, pooled_features_2)
        score = torch.sigmoid(soft_matrix.sum() + bias_value)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        # score为预测的ged值，pre_ged.item()为处理过的预测ged值
        return score, pre_ged.item(), map_matrix, masked_index


class MyGNN2(torch.nn.Module):

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(MyGNN2, self).__init__()
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
        self.lg_mapMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)
        # self.costMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)
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
        
        # my, edge gin
        lg_nn1 = torch.nn.Sequential(
            torch.nn.Linear(self.number_labels, self.args.filters_1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.filters_1, self.args.filters_1),
            torch.nn.BatchNorm1d(self.args.filters_1, track_running_stats=False))
        lg_nn2 = torch.nn.Sequential(
            torch.nn.Linear(self.args.filters_1, self.args.filters_2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.filters_2, self.args.filters_2),
            torch.nn.BatchNorm1d(self.args.filters_2, track_running_stats=False))
        lg_nn3 = torch.nn.Sequential(
            torch.nn.Linear(self.args.filters_2, self.args.filters_3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.filters_3, self.args.filters_3),
            torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))
        self.lg_c1 = GINConv(lg_nn1, train_eps=True)
        self.lg_c2 = GINConv(lg_nn2, train_eps=True)
        self.lg_c3 = GINConv(lg_nn3, train_eps=True)
        # #
        self.lg_attention = AttentionModule(self.args)
        self.lg_tensor_network = TensorNetworkModule(self.args)
        self.lg_fully_connected_first = torch.nn.Linear(self.args.tensor_neurons,
                                                     self.args.bottle_neck_neurons)
        self.lg_fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.lg_fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.lg_scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)
        
        # LRL模块
        self.transform1 = torch.nn.Linear(self.args.filters_3, 64)
        self.relu1 = torch.nn.ReLU()
        self.transform2 = torch.nn.Linear(64, 64)
        
        self.lg_transform1 = torch.nn.Linear(self.args.filters_3, 64)
        self.lg_relu1 = torch.nn.ReLU()
        self.lg_transform2 = torch.nn.Linear(64, 64)
        
        # lg + simgnn gcn
        # self.feature_count = self.args.tensor_neurons + self.args.bins
        # self.lg_convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        # self.lg_convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        # self.lg_convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        # bias
        # self.lg_attention = AttentionModule(self.args)
        # self.lg_tensor_network = TensorNetworkModule(self.args)
        # self.lg_fully_connected_first = torch.nn.Linear(self.feature_count,
        #                                              self.args.bottle_neck_neurons)
        # self.lg_fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
        #                                               self.args.bottle_neck_neurons_2)
        # self.lg_fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
        #                                              self.args.bottle_neck_neurons_3)
        # self.lg_scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)
        
    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()  # n*m
        scores = scores.view(-1, 1)  # nm*1
        hist = torch.histc(scores, bins=self.args.bins)  # 16
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)  # 1*16
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

    def lg_simgnn_gcn(self, edge_index, features):
        features = self.lg_convolution_1(features, edge_index)
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
    
    def lg_convolutional(self, lg_edge_index, features):
        """边图的图卷积网络
        Args:
            lg_edge_index (list): 边索引[((1,2),(3,4)),((,),(,)),...]
            features (torch.Tensor): 节点特征向量 n*29
        Returns:
            _type_: 节点嵌入 n*32
        """
        features = self.lg_c1(features, lg_edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.lg_c2(features, lg_edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.lg_c3(features, lg_edge_index)
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
    
    def lg_get_bias_value(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.lg_attention(abstract_features_1)
        pooled_features_2 = self.lg_attention(abstract_features_2)
        scores = self.lg_tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.lg_fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.lg_fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.lg_fully_connected_third(scores))
        score = self.lg_scoring_layer(scores).view(-1)
        return score

    def lg_simgnn(self, abstract_features_1, abstract_features_2):
        # NTN
        pooled_features_1 = self.lg_attention(abstract_features_1)  # 32*1
        pooled_features_2 = self.lg_attention(abstract_features_2)
        scores = self.lg_tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)
        # hist
        hist = self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))  # 1*16
        scores = torch.cat((scores, hist), dim=1).view(1, -1)  # 1*16
        scores = F.relu(self.lg_fully_connected_first(scores))
        scores = F.relu(self.lg_fully_connected_second(scores))
        scores = F.relu(self.lg_fully_connected_third(scores))
        
        score = torch.sigmoid(self.scoring_layer(scores).view(-1))
        return score
    
    @staticmethod
    def ged_from_mapping(matrix, A1, A2, f1, f2):
        # edge loss
        A_loss = torch.mm(torch.mm(matrix.t(), A1), matrix) - A2
        # label loss
        F_loss = torch.mm(matrix.t(), f1) - f2
        mapping_ged = ((A_loss * A_loss).sum() + (F_loss * F_loss).sum()) / 2.0
        return mapping_ged.view(-1)

    def node_alignment_with_edge(self, map_matrix, n1, n2, lg_map_matrix, lg_node_list_1, lg_node_list_2):
        """
        Args:
            map_matrix (torch.Tensor): 节点相似度 max_size*max_size
            lg_map_matrix (torch.Tensor): 边匹配矩阵 g*h 取值[-1,1]之间
            lg_node_list_1 (list): 边图1节点列表 g
            lg_node_list_2 (list): 边图2节点列表 h
        Returns:
            map_matrix (torch.Tensor): 节点对齐结果 n
        """
        # max_size*max_size -> n1*n2
        map_matrix = -map_matrix  # 因为是cost矩阵需要取反
        lg_map_matrix = -lg_map_matrix
        max_size = map_matrix.shape[0]
        matrix = map_matrix[:n1, :n2]  # n1*n2
        # print("matrix.shape =", matrix.shape)  # zhj
        # print("matrix: \n", matrix)
        # print("lg_map_matrix.shape =", lg_map_matrix.shape)
        # print("lg_map_matrix: \n", lg_map_matrix)
        # print(len(lg_node_list_1), len(lg_node_list_2))
        # print(lg_node_list_1)
        # print(lg_node_list_2)
        # print("lg_map_matrix:\n", lg_map_matrix)
        # 通过节点相似度矩阵和边相似度矩得到节点对齐结果
        aligment_index = []
        n, m = matrix.shape
        x = min(n, m)  # 最多对齐的节点数
        i = 0
        while i < x:
            """ 1、从节点相似度矩阵中得到节点对x """
            max_index = torch.argmax(matrix).item()
            max_row = max_index // m
            max_col = max_index % m
            # print("原图节点对x:", max_row, max_col)  # zhj
            aligment_index.append([max_row, max_col])
            i += 1
            if i == x: break
            matrix[max_row, :] = -100
            matrix[:, max_col] = -100
            """ 2、根据节点对x, 从边相似度矩阵找到另一节点对y """
            row_list = []  # 在边相似度矩阵中，与得到“原图节点”索引相关的“边图节点”索引
            col_list = []
            for index in range(len(lg_node_list_1)):
                if max_row in lg_node_list_1[index]: row_list.append(index)
            for index in range(len(lg_node_list_2)):
                if max_col in lg_node_list_2[index]: col_list.append(index)
            # 得到边图节点对z
            max_score = -100
            lg_node_1, lg_node_2 = (), ()
            for j in row_list:  # 遍历边相似度度矩阵中“原图节点”相关的元素，选择最大值
                for k in col_list:
                    if lg_map_matrix[j][k] > max_score:
                        max_score = lg_map_matrix[j][k]
                        lg_node_1 = lg_node_list_1[j]
                        lg_node_2 = lg_node_list_2[k]
            # print("边图节点对z:", lg_node_1, lg_node_2)  # zhj
            if lg_node_1 == () or lg_node_2 == ():
                print("lg_map_matrix: \n", lg_map_matrix)
                print(lg_node_list_1)
                print(lg_node_list_2)
                print("原图节点对x:", max_row, max_col)
                print("row_list:", row_list)
                print("col_list:", col_list)
                print(j, k)
                print("边图节点对z:", lg_node_1, lg_node_2)
            # 从边图节点对z中得到原图节点对y
            new_row = lg_node_1[0] if max_row == lg_node_1[1] else lg_node_1[1]
            new_col = lg_node_2[0] if max_col == lg_node_2[1] else lg_node_2[1]
            # print("原图节点对y:", new_row, new_col)  # zhj
            """ 3、根据节点相似度矩阵的情况, 判断节点对y是否舍弃 """
            if matrix[new_row, new_col] != -100:
                # print("1")
                aligment_index.append([new_row, new_col])
                i += 1
                matrix[new_row, :] = -100
                matrix[:, new_col] = -100
            # print("------------------------")
        # print("节点对齐结果:", aligment_index)  # zhj
        aligment_matrix = torch.zeros(max_size, max_size)
        for index in aligment_index:
            aligment_matrix[index[0], index[1]] = 1
        # print("aligment_matrix.shape =", aligment_matrix.shape)  # zhj
        # print("节点对齐矩阵:\n", aligment_matrix)  # zhj
        bias_matrix = torch.mul(aligment_matrix, map_matrix)
        return bias_matrix

    def Euclidean_Distance(self, abstract_features_1, abstract_features_2):
        """计算两张图节点嵌入间的欧式距离
        Args:
            abstract_features_1 (torch.Tensor): 图1的节点嵌入 n*d
            abstract_features_2 (torch.Tensor): 图2的节点嵌入 m*d
        Returns:
            torch.Tensor: 距离矩阵 n*m
        """
        # 扩展两个张量以进行向量化距离计算
        # abstract_features_1.unsqueeze(1) 将变成 n x 1 x d
        # abstract_features_2.unsqueeze(0) 将变成 1 x m x d
        # 结果将会广播成 n x m x 32
        diff = abstract_features_1.unsqueeze(1) - abstract_features_2.unsqueeze(0)
        cost_matrix = torch.sqrt(torch.sum(diff ** 2, dim=2))
        # 计算最小值和最大值
        min_val = torch.min(cost_matrix)
        max_val = torch.max(cost_matrix)
        # 进行最小-最大归一化
        cost_matrix = (cost_matrix - min_val) / (max_val - min_val)
        # 归一化的值范围[-1,1]
        # cost_matrix = cost_matrix * 2 -1
        return cost_matrix

    def LRL(self, abstract_features_1, abstract_features_2, flag):
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
        # 填充
        abstract_features_1 = F.pad(abstract_features_1, pad=(0,0,0,max_size-n1))  # [max_size, 32]
        abstract_features_2 = F.pad(abstract_features_2, pad=(0,0,0,max_size-n2))  # [max_size, 32]
        # LRL
        if flag:  # 节点
            transformed_node_emb_1 = self.transform2(self.relu1(self.transform1(abstract_features_1)))  # [max_size, 64]
            transformed_node_emb_2 = self.transform2(self.relu1(self.transform1(abstract_features_2)))  # [max_size, 64]
        else:  # 边
            transformed_node_emb_1 = self.lg_transform2(self.lg_relu1(self.lg_transform1(abstract_features_1)))  # [max_size, 64]
            transformed_node_emb_2 = self.lg_transform2(self.lg_relu1(self.lg_transform1(abstract_features_2)))  # [max_size, 64]
        # mask
        dim = transformed_node_emb_1.shape[1]
        mask1 = torch.cat((torch.tensor([1]).repeat(n1,1).repeat(1,dim), torch.tensor([0]).repeat(max_size-n1,1).repeat(1,dim)))  # [max_size, 64]
        mask2 = torch.cat((torch.tensor([1]).repeat(n2,1).repeat(1,dim), torch.tensor([0]).repeat(max_size-n2,1).repeat(1,dim)))  # [max_size, 64]
        emb_1 = torch.mul(mask1, transformed_node_emb_1)  # 逐元素相乘
        emb_2 = torch.mul(mask2, transformed_node_emb_2)
        # 虽然在上面mask掉了填充部分，但经过gumbel后，mask的部分仍会有值，需要A_match方面进行mask
        return torch.matmul(emb_1, emb_2.permute(1,0))

    def Cross(self, abstract_features_1, abstract_features_2, flag):
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
        if flag:  # 节点
            m = self.mapMatrix(abstract_features_1, abstract_features_2)  # [max_size, max_size]
        else:   # 边
            m = self.lg_mapMatrix(abstract_features_1, abstract_features_2)  # [max_size, max_size]
        # mask
        if n1 > n2:
            mask = torch.zeros(max_size, max_size)
            mask[:, :n2] = 1
            return torch.mul(mask, m)
        elif n1 < n2:
            mask = torch.cat((torch.tensor([1]).repeat(n1,1).repeat(1,max_size), torch.tensor([0]).repeat(max_size-n1,1).repeat(1,max_size)))  # [max_size, max_size]
            return torch.mul(mask, m)
        return m 

    def lg_cross(self, f1, f2):
        m = torch.matmul(f1, f2.permute(1,0))
        x_min = m.min()
        x_max = m.max()
        if x_max - x_min == 0:  # 只有一个元素或者所有元素相同
            print(m)
            m.zero_()  # 所有元素归0
            return m
        m_normalized = (m - x_min) / (x_max - x_min)  # 缩放到 [0, 1]
        m_scaled = 2 * m_normalized - 1  # 缩放到 [-1, 1]
        return m_scaled

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :param is_testing: whether return ged value together with ged score
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]  # (torch.Tensor)2*n
        edge_index_2 = data["edge_index_2"]  # (torch.Tensor)2*m
        features_1 = data["features_1"]  # (torch.Tensor)n*d
        features_2 = data["features_2"]  # (torch.Tensor)m*d
        
        lg_node_list_1 = data["lg_node_list_1"]  # (list)边图的节点列表：[(7, 3), (3, 8), (3, 9),...]
        lg_node_list_2 = data["lg_node_list_2"]
        lg_edge_index_mapping_1 = data["lg_edge_index_mapping_1"]  # 边图的边索引映射[((7, 3), (3, 9)),...]->[(0, 2),...]
        lg_edge_index_mapping_2 = data["lg_edge_index_mapping_2"]
        lg_features_1 = data["lg_features_1"]
        lg_features_2 = data["lg_features_2"]
        
        # 计算节点嵌入
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        
        # 计算边嵌入
        flag_1 = 0
        flag_2 = 0
        if len(lg_node_list_1) == 1: 
            flag_1 = 1
            lg_features_1 = torch.cat([lg_features_1, lg_features_1], dim=0)
        if len(lg_node_list_2) == 1: 
            flag_2 = 1
            lg_features_2 = torch.cat([lg_features_2, lg_features_2], dim=0)
        lg_abstract_features_1 = self.lg_convolutional(lg_edge_index_mapping_1, lg_features_1)
        # lg_abstract_features_1 = self.lg_simgnn_gcn(lg_edge_index_mapping_1, lg_features_1)
        lg_abstract_features_2 = self.lg_convolutional(lg_edge_index_mapping_2, lg_features_2)
        # lg_abstract_features_2 = self.lg_simgnn_gcn(lg_edge_index_mapping_2, lg_features_2)
        if flag_1: lg_abstract_features_1 = lg_abstract_features_1[0:1]
        if flag_2: lg_abstract_features_2 = lg_abstract_features_2[0:1]
            

        # 计算节点和边相似度矩阵，以及节点的成本矩阵
        # cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)  # n*m
        # cost_matrix = self.Euclidean_Distance(abstract_features_1, abstract_features_2)
        # map_matrix = self.mapMatrix(abstract_features_1, abstract_features_2)  # n*m
        map_matrix = self.Cross(abstract_features_1, abstract_features_2, 1)  # [max_size, max_size]
        lg_map_matrix = self.Cross(lg_abstract_features_1, lg_abstract_features_2, 0)  # [e_max_size, e_max_size]
        # print("lg_features_1.shape =", lg_features_1.shape)
        # print("lg_features_2.shape =", lg_features_2.shape)
        # print("lg_abstract_features_1 =", lg_abstract_features_1.shape)
        # print("lg_abstract_features_2 =", lg_abstract_features_2.shape)
        # lg_map_matrix = self.lg_cross(lg_abstract_features_1, lg_abstract_features_2)  # en*em
        # print("lg_map_matrix =", lg_map_matrix.shape)
        
        # print(" ")  # zhj
        # print("n1 =", data["n1"])
        # print("n2 =", data["n2"])
        # print("edge_index_1.shape =", edge_index_1.shape)
        # print("edge_index_2.shape =", edge_index_2.shape)
        # print("features_1.shape =", features_1.shape)
        # print("features_2.shape =", features_2.shape)
        # print("lg_n1 =", len(lg_node_list_1))
        # print("lg_n2 =", len(lg_node_list_2))
        # print("lg_edge_index_mapping_1.shape =", lg_edge_index_mapping_1.shape)
        # print("lg_edge_index_mapping_2.shape =", lg_edge_index_mapping_2.shape)
        # print("lg_features_1.shape =", lg_features_1.shape)
        # print("lg_features_2.shape =", lg_features_2.shape)
        # print(" ")
        
        LRL_map_matrix = self.LRL(abstract_features_1, abstract_features_2, 1)  # [max_size, max_size]
        lg_LRL_map_matrix = self.LRL(lg_abstract_features_1, lg_abstract_features_2, 0)  # [e_max_size, e_max_size]
        
        bias_matrix = self.node_alignment_with_edge(map_matrix, data["n1"], data["n2"], lg_map_matrix, lg_node_list_1, lg_node_list_2)  # [max_size, max_size]
        # print("---------------------------------------------------")
        node_alignment = gumbel_sinkhorn(LRL_map_matrix, tau=0.1, noise=True, bias=bias_matrix)  # [max_size, max_size]
        # node_alignment = gumbel_sinkhorn(LRL_map_matrix, tau=0.1)  # [max_size, max_size]
        lg_node_alignment = gumbel_sinkhorn(lg_LRL_map_matrix, tau=0.1)
        
        # print(map_matrix.shape)
        # print(map_matrix)
        # print(node_alignment.shape)
        # print(node_alignment)
        # print("------------------------------------------------------------------------\n\n")
        # time.sleep(3)

        # 计算map_matrix和bias
        # m = torch.nn.Softmax(dim=1)  # 是否要用Softmax
        # soft_matrix = map_matrix * cost_matrix
        soft_matrix = map_matrix * node_alignment
        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2)
        score = torch.sigmoid(soft_matrix.sum() + bias_value)
        
        lg_soft_matrix = lg_map_matrix * lg_node_alignment
        lg_bias_value = self.lg_get_bias_value(lg_abstract_features_1, lg_abstract_features_2)
        lg_score = torch.sigmoid(lg_soft_matrix.sum() + lg_bias_value)
        
        # lg_score = self.lg_simgnn(lg_abstract_features_1, lg_abstract_features_2)
        
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
            
        else:
            assert False
        return score, pre_ged.item(), lg_score
        # return score, pre_ged.item(), map_matrix  # gedgnn


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
        self.costMatrix = GedMatrixModule(self.args.filters_2, self.args.hidden_dim)

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
        self.transform1 = torch.nn.Linear(self.args.filters_3, 64)
        self.relu1 = torch.nn.ReLU()
        self.transform2 = torch.nn.Linear(64, 64)
        
        # 
        self.transform = torch.nn.Linear(self.args.filters_3, self.number_labels)  # 32->29
        
        # AReg
        self.trans = torch.nn.Linear(self.args.filters_3, self.args.filters_3)  # 32->32
        self.gamma = torch.nn.Parameter(torch.Tensor(1))
        
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
        # 填充
        abstract_features_1 = F.pad(abstract_features_1, pad=(0,0,0,max_size-n1))  # [max_size, 32]
        abstract_features_2 = F.pad(abstract_features_2, pad=(0,0,0,max_size-n2))  # [max_size, 32]
        # LRL
        transformed_node_emb_1 = self.transform2(self.relu1(self.transform1(abstract_features_1)))  # [max_size, 64]
        transformed_node_emb_2 = self.transform2(self.relu1(self.transform1(abstract_features_2)))  # [max_size, 64]
        # mask
        dim = transformed_node_emb_1.shape[1]
        mask1 = torch.cat((torch.tensor([1]).repeat(n1,1).repeat(1,dim), torch.tensor([0]).repeat(max_size-n1,1).repeat(1,dim))).to(self.device)  # [max_size, 64]
        mask2 = torch.cat((torch.tensor([1]).repeat(n2,1).repeat(1,dim), torch.tensor([0]).repeat(max_size-n2,1).repeat(1,dim))).to(self.device)  # [max_size, 64]
        emb_1 = torch.mul(mask1, transformed_node_emb_1)  # 逐元素相乘
        emb_2 = torch.mul(mask2, transformed_node_emb_2)
        # 虽然在上面mask掉了填充部分，但经过gumbel后，mask的部分仍会有值，需要A_match方面进行mask
        return torch.matmul(emb_1, emb_2.permute(1,0))
    
    def Cross(self, abstract_features_1, abstract_features_2):
        """通过嵌入计算相似度矩阵
        Args:
            abstract_features_1 (_type_): 图1嵌入 [n1, 32]
            abstract_features_2 (_type_): 图2嵌入 [n2, 32]
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
        m = self.costMatrix(abstract_features_1, abstract_features_2)  # [max_size, max_size]
        # m = F.pad(m, pad=(0,0,0,max_size-n1))
        # mask
        if n1 > n2:
            mask = torch.zeros(max_size, max_size).to(self.device)
            mask[:, :n2] = 1
            return torch.mul(mask, m)
        elif n1 < n2:
            # 为什么不使用上面生成mask的方法？？？
            mask = torch.cat((torch.tensor([1]).repeat(n1,1).repeat(1,max_size), torch.tensor([0]).repeat(max_size-n1,1).repeat(1,max_size))).to(self.device)  # [max_size, max_size]
            return torch.mul(mask, m)
        return m 
    
    def generate_pseudo_graph(self, LRL_map_matrix, node_alignment, features_1, features_2):
        """根据节点相似度矩阵和节点对齐结果交换图1和图2中相似度前60%节点的特征或嵌入

        Args: n1<=n2
            LRL_map_matrix (_type_): 节点相似度矩阵 n2*n2
            node_alignment (_type_): 节点对齐结果 n2*n2
            features_1 (_type_): 图1节点特征 n1*29
            features_2 (_type_): 图2节点特征 n2*29

        Returns:
            _type_: 交换后的节点特征
        """
        n1 = features_1.shape[0]
        # 转化为0-1矩阵
        binarized_matrix = (node_alignment >= 0.5).float()
        # 裁剪成n1*n2大小
        cropped_matrix = binarized_matrix[:n1, :]  # 数据集中图1应该都是小于图2的
        # 获取所有元素为 1 的索引tensor([[x,y],[x,y],...])
        indices = torch.nonzero(cropped_matrix)  
        value = torch.tensor([LRL_map_matrix[i[0], i[1]] for i in indices])  # 索引在相似度矩阵的对应的元素
        k = int(len(value) * 0.6)
        if k < 1:  # node_alignment中某些行可能不存在大于0.5的值
            return features_1, features_2, 0.0
        # 获取前k个最大元素的值及其原索引
        _, value_topk_indices = torch.topk(value, k)  # tensor([,,])
        # 交换特征
        pseudo_features_1 = features_1.clone()
        pseudo_features_2 = features_2.clone()
        for x in value_topk_indices:
            indice = indices[x]  # 通过value索引获得对应的索引
            i = indice[0]
            j = indice[1]
            pseudo_features_1[i] = features_2[j]
            pseudo_features_2[j] = features_1[i]
        
        return pseudo_features_1, pseudo_features_2, k/n1

    def Reg2(self, n1, n2, G1, G2):
        """ 利用正则化进行细粒度交互，按照ERIC的论文
            D(*,*)为cosin相似度
        Args:
            n1 (_type_): n1 * filters_3
            n2 (_type_): n2 * filters_3
            G1 (_type_): filters_3 * 1
            G2 (_type_): filters_3 * 1
        """
        # Y1 = || D(N1,G1) - D(N1,G2) ||2
        g1 = torch.t(G1).repeat(n1.size(0), 1)  # n1*filters_3
        g2 = torch.t(G2).repeat(n1.size(0), 1)  
        D_n1_g1 = F.cosine_similarity(n1, g1, dim=1, eps=1e-8)  # 1*n1
        D_n1_g2 = F.cosine_similarity(n1, g2, dim=1, eps=1e-8)
        Y1 = torch.norm(D_n1_g1 - D_n1_g2, dim=0)  # tensor([x])  # 行的维度为1，使用dim=0
        # Y2 = || D(N2,G1) - D(N2,G2) ||2
        g1 = torch.t(G1).repeat(n2.size(0), 1)  # n1*filters_3
        g2 = torch.t(G2).repeat(n2.size(0), 1)  
        D_n2_g1 = F.cosine_similarity(n2, g1, dim=1, eps=1e-8)  # 1*n1
        D_n2_g2 = F.cosine_similarity(n2, g2, dim=1, eps=1e-8)
        Y2 = torch.norm(D_n2_g1 - D_n2_g2, dim=0)  # tensor([x])
        # L = Y1 + Y2 + || Y1 -Y2 ||2  单值的第二范数是为了取绝对值吗
        Loss = Y1 + Y2 + torch.norm(Y1 - Y2, dim=0)
        
        return Loss

    def Reg(self, n1, n2, G1, G2):
        """ 使图1的节点嵌入与图1嵌入比与图2嵌入更近
            D(N1, G1) < D(N1, G2)   
            D(N2, G2) < D(N2, G1)
            使用三元组损失
            L1 = [D(N1, G1) - D(N1, G2) + margin]+
            L2 = [D(N2, G2) - D(N2, G1) + margin]+
        Args:
            n1 (_type_): n1 * filters_3
            n2 (_type_): n2 * filters_3
            G1 (_type_): filters_3 * 1
            G2 (_type_): filters_3 * 1
        """
        
        g1 = torch.t(G1).repeat(n1.size(0), 1)  # n1*filters_3
        g2 = torch.t(G2).repeat(n1.size(0), 1)  
        D_n1_g1 = F.cosine_similarity(n1, g1, dim=1, eps=1e-8)  # 1*n1
        D_n1_g2 = F.cosine_similarity(n1, g2, dim=1, eps=1e-8)
        # D_n1_g1 = torch.norm(n1 - g1, dim=1)  # 1*n1
        # D_n1_g2 = torch.norm(n1 - g2, dim=1)  
        loss_1 = F.relu(D_n1_g2 - D_n1_g1 + 0.2)  # 1*n1  # 余弦相似度需反转
        
        g1 = torch.t(G1).repeat(n2.size(0), 1)  # n2*filters_3
        g2 = torch.t(G2).repeat(n2.size(0), 1)
        D_n2_g2 = F.cosine_similarity(n2, g2, dim=1, eps=1e-8)  # 1*n2
        D_n2_g1 = F.cosine_similarity(n2, g1, dim=1, eps=1e-8)
        # D_n2_g2 = torch.norm(n2 - g2, dim=1)  # 1*n2
        # D_n2_g1 = torch.norm(n2 - g1, dim=1)
        loss_2 = F.relu(D_n2_g1 - D_n2_g2 + 0.2)
        # print(n1)
        # print(G1)
        # print(n2)
        # print(G2)
        # print("++++++++++++++++++++++++++++++++++++++++")
        # print(D_n1_g1)
        # print(D_n1_g2)
        # print(loss_1)
        # print(loss_2)
        # print("-----------------------------------------")
        # time.sleep(1)
        
        return loss_1.mean() + loss_2.mean()
    
    def Euclidean_Distance(self, abstract_features_1, abstract_features_2):
        """计算节点成本矩阵
        Args:
            abstract_features_1 (torch.Tensor): 图1的节点嵌入 n1*d
            abstract_features_2 (torch.Tensor): 图2的节点嵌入 n2*d
        Returns:
            torch.Tensor: 成本矩阵 n*m
        """
        # 扩展两个张量以进行向量化距离计算
        # abstract_features_1.unsqueeze(1) 将变成 n1 x 1 x d
        # abstract_features_2.unsqueeze(0) 将变成 1 x n2 x d
        # 结果将会广播成 n1 x n2 x 32
        diff = abstract_features_1.unsqueeze(1) - abstract_features_2.unsqueeze(0)
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=2))  # n1*n2
        # 计算最小值和最大值
        # min_val = torch.min(cost_matrix)
        # max_val = torch.max(cost_matrix)
        # 进行最小-最大归一化
        # cost_matrix = (cost_matrix - min_val) / (max_val - min_val)
        # 归一化的值范围[-1,1]
        # cost_matrix = cost_matrix * 2 -1
        # 填充成n2*n2方阵(n1<n2)
        n1 = abstract_features_1.size(0)
        n2 = abstract_features_2.size(0)
        cost_matrix = torch.zeros((n2, n2), dtype=torch.float32)
        cost_matrix[:n1, :n2] = dist_matrix
        # print(cost_matrix)
        
        return cost_matrix
    
    def Cosin_similarty(self, f1, f2):
        """ 计算节点成本矩阵"""
        # 定义一个小常数避免除以零
        eps = 1e-8
        # 计算范数，除以范数
        f1_norm = f1 / (f1.norm(dim=1, keepdim=True) + eps)  # n1*d
        f2_norm = f2 / (f2.norm(dim=1, keepdim=True) + eps)  # n2*d
        
        # 计算余弦相似度矩阵
        cos_similarity_matrix = torch.matmul(f1_norm, f2_norm.t())  # n1*n2
        # 填充成n2*n2方阵(n1<n2)
        n1 = f1.size(0)
        n2 = f2.size(0)
        cost_matrix = torch.zeros((n2, n2), dtype=torch.float32)
        cost_matrix[:n1, :n2] = -cos_similarity_matrix
        
        return cost_matrix
    
    def get_positive_expectation(self, p_samples):
        """Computes the positive part of a divergence / difference.
        """
        log_2 = math.log(2.)
        Ep = log_2 - F.softplus(- p_samples)
        return Ep
    
    def AReg(self, f_1, f_2):
        # f1 -> Linear(d,d) -> f1'
        # f_1 = torch.nn.functional.relu(self.trans(f_1))  # mlp
        # f_2 = torch.nn.functional.relu(self.trans(f_2))  # mlp
        f_1 = torch.nn.functional.relu(f_1)
        f_2 = torch.nn.functional.relu(f_2)
        batch_1 = torch.zeros(f_1.size(0), dtype=torch.int64).to(self.device)
        batch_2 = torch.zeros(f_2.size(0), dtype=torch.int64).to(self.device)
        g_1 = global_add_pool(f_1, batch_1)
        g_2 = global_add_pool(f_2, batch_2)
        self_sim_1   = torch.mm(f_1, g_1.t())  # n1*1
        self_sim_2   = torch.mm(f_2, g_2.t())  # n2*1
        cross_sim_12 = torch.mm(f_1, g_2.t())  # n1*1
        cross_sim_21 = torch.mm(f_2, g_1.t())  # n2*1
        L_1 = self.get_positive_expectation(self_sim_1).sum()- self.get_positive_expectation(cross_sim_12).sum()
        L_2 = self.get_positive_expectation(self_sim_2).sum()- self.get_positive_expectation(cross_sim_21).sum()
        
        
        # print(f_1)
        # print(f_2)
        # print("-----------------------------------------")
        # print(self_sim_1)
        # print(self_sim_2)
        # print("-----------------------------------------")
        # print(cross_sim_12)
        # print(cross_sim_21)
        # print("-----------------------------------------")
        # print(g_1)
        # print(g_2)
        # print(L_1, L_2, L_1 - L_2)
        # print("=========================================")
        # time.sleep(2)
        return L_1 - L_2
    
    def LRL_Cross(self, f1, f2):
        n1 = f1.shape[0]
        n2 = f2.shape[0]
        max_n = max(n1, n2)
        max_size = 10
        # LRL
        emb_1 = self.transform2(self.relu1(self.transform1(f1)))  # [n1, 32]
        emb_2 = self.transform2(self.relu1(self.transform1(f2)))  # [n2, 32]
        # emb_1 = F.pad(emb_1, pad=(0,0,0,max_n-n1))  # [max_n, 32]
        # emb_2 = F.pad(emb_2, pad=(0,0,0,max_n-n2))  # [max_n, 32]
        # Cross
        cost_matrix = self.costMatrix(emb_1, emb_2)  # [n1, n2]  注意costMatrix的参数
        # gs
        sinkhorn_input = F.pad(cost_matrix, pad=(0,max_n-n2,0,max_n-n1))  # [max_n, max_n]
        transport_plan = gumbel_sinkhorn(sinkhorn_input, tau=0.1)  # [max_n, max_n]
        # 填充
        transport_plan = F.pad(transport_plan, pad=(0,max_size-max_n,0,max_size-max_n))  # [10, 10]
        cost_matrix = F.pad(cost_matrix, pad=(0,max_size-n2,0,max_size-n1))  # [10,10]
        return cost_matrix, transport_plan
    
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
        
        # 计算节点嵌入
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        
        cost_matrix, node_alignment = self.LRL_Cross(abstract_features_1, abstract_features_2)
        
        # 计算节点成本矩阵
        # cost_matrix = self.Cross(abstract_features_1, abstract_features_2)  # max*max mask (max-n)*(max-m)
        # cost_matrix = self.Euclidean_Distance(abstract_features_1, abstract_features_2)
        # cost_matrix = self.Cosin_similarty(abstract_features_1, abstract_features_2)
        
        # 计算节点对齐矩阵
        # LRL_map_matrix = self.LRL(abstract_features_1, abstract_features_2)  # max*max
        # node_alignment = gumbel_sinkhorn(LRL_map_matrix, tau=0.1)  # # max*max
        
        
        # 1 相似度矩阵
        # pseudo_features_1, pseudo_features_2 = self.generate_pseudo_graph(LRL_map_matrix, node_alignment, features_1, features_2)
        # f_1 = self.convolutional_pass(edge_index_1, pseudo_features_1)
        # f_2 = self.convolutional_pass(edge_index_2, pseudo_features_2)
        # pseudo_map_matrix = self.LRL(f_1, f_2)  # max*max
        # n = f_1.size(0)
        # LRL_map_matrix = LRL_map_matrix[:n, :]
        # pseudo_map_matrix = pseudo_map_matrix[:n, :]
        
        # 2 成本矩阵
        # pseudo_features_1, pseudo_features_2, ratio= self.generate_pseudo_graph(LRL_map_matrix, node_alignment, abstract_features_1, abstract_features_2)
        # pseudo_features_1 = self.transform(pseudo_features_1)  # n*32 -> n*29
        # pseudo_features_2 = self.transform(pseudo_features_2)
        # # ont-hot Feature和交换后的embedding加权叠加
        # pseudo_features_1 = (pseudo_features_1 + features_1)/2
        # pseudo_features_2 = (pseudo_features_2 + features_2)/2
        # # 输入到GNN中
        # f_1 = self.convolutional_pass(edge_index_1, pseudo_features_1)
        # f_2 = self.convolutional_pass(edge_index_2, pseudo_features_2)
        # pseudo_cost_matrix = self.Cross(f_1, f_2)  # max*max
        # # 裁减掉0值
        # n = f_1.size(0)
        # matrix1 = cost_matrix[:n, :]
        # matrix2 = pseudo_cost_matrix[:n, :]
        
        # print("pseudo_features_1.shape =", pseudo_features_1.shape)
        # print("pseudo_features_2.shape =", pseudo_features_2.shape)
        # print("f_1.shape =", f_1.shape)
        # print("f_2.shape =", f_2.shape)
        # print("pseudo_cost_matrix.shape =", pseudo_cost_matrix.shape)
        # print(pseudo_cost_matrix)
        # print("------------------------------------------")
        # time.sleep(2)
     
        # 计算map_matrix和bias
        # m = torch.nn.Softmax(dim=1)
        soft_matrix = node_alignment * cost_matrix
        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2)
        score = torch.sigmoid(soft_matrix.sum() + bias_value)  # 0~1
        
        # Reg
        # loss_reg = self.Reg(abstract_features_1, abstract_features_2, g1, g2)  # 欧式或余弦
        # loss_reg = self.Reg2(abstract_features_1, abstract_features_2, g1, g2)  # 
        
        # loss_reg = 0
        # if self.training:
        #     loss_reg = self.AReg(abstract_features_1, abstract_features_2) * self.gamma
        
        # print(abstract_features_1)
        # print(abstract_features_2)
        # print("-----------------------------------------")
        # print(node_alignment)
        # print(cost_matrix)
        # print(soft_matrix)
        # print("-----------------------------------------")
        # print(bias_value)
        # print(score)
        # print(loss_reg)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # time.sleep(2)
        
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False

        
        
        # return score, pre_ged.item(), matrix1, matrix2, ratio
        # return score, pre_ged.item(), loss_reg, self.gamma
        return score, pre_ged.item(), cost_matrix


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