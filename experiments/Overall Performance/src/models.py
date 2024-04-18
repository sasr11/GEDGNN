import dgl
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCNConv, GINConv
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
    
    def transport_plan(self, m):
        """根据嵌入, 通过gumbel_sinkhorn算法计算对齐结果
        Args:
            ne1: 图1节点嵌入 n*d
            ne2: 图2节点嵌入 n*d
        """
        # print(m.shape)
        # input = torch.matmul(ne1, ne2.permute(1,0))
        return gumbel_sinkhorn(m, 0.1)
    
    def lg_convolutional(self, lg_edge_index, features):
        """边图的图卷积网络
        Args:
            lg_edge_index (list): 边索引[((1,2),(3,4)),((,),(,)),...]
            features (torch): 节点特征向量
        Returns:
            _type_: 节点嵌入
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
        # features = torch.sigmoid(features)
        return features

    def node_alignment_with_edge(self, map_matrix, lg_map_matrix, lg_node_list_1, lg_node_list_2):
        # 通过节点相似度矩阵和边相似度矩得到节点对齐结果
        aligment_index = []
        n, m = map_matrix.shape
        x = min(n, m)  # 最多对齐的节点数
        i = 0
        while i < x:
            """ 1、从节点相似度矩阵中得到节点对x """
            max_index = torch.argmax(map_matrix).item()
            max_row = max_index // m
            max_col = max_index % m
            # print("原图节点对x:", max_row, max_col)
            aligment_index.append([max_row, max_col])
            i += 1
            if i == x: break
            map_matrix[max_row, :] = -1
            map_matrix[:, max_col] = -1
            """ 2、根据节点对x, 从边相似度矩阵找到另一节点对y """
            row_list = []  # 在边相似度矩阵中，与得到“原图节点”索引相关的“边图节点”索引
            col_list = []
            for index in range(len(lg_node_list_1)):
                if max_row in lg_node_list_1[index]: row_list.append(index)
            for index in range(len(lg_node_list_2)):
                if max_col in lg_node_list_2[index]: col_list.append(index)
            # 得到边图节点对z
            max_score = -1
            lg_node_1, lg_node_2 = None, None
            for j in row_list:  # 遍历边相似度度矩阵中“原图节点”相关的元素，选择最大值
                for k in col_list:
                    if lg_map_matrix[j][k] > max_score:
                        max_score = lg_map_matrix[j][k]
                        lg_node_1 = lg_node_list_1[j]
                        lg_node_2 = lg_node_list_2[k]
            # print("边图节点对z:", lg_node_1, lg_node_2)
            # 从边图节点对z中得到原图节点对y
            new_row = lg_node_1[0] if max_row == lg_node_1[1] else lg_node_1[1]
            new_col = lg_node_2[0] if max_col == lg_node_2[1] else lg_node_2[1]
            # print("原图节点对y:", new_row, new_col)
            """ 3、根据节点相似度矩阵的情况, 判断节点对y是否舍弃 """
            if map_matrix[new_row, new_col] != -1:
                aligment_index.append([new_row, new_col])
                i += 1
                map_matrix[new_row, :] = -1
                map_matrix[:, new_col] = -1
        
        map_matrix.zero_()
        for index in aligment_index:
            map_matrix[index[0], index[1]] = 1
        return map_matrix

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
        
        lg_node_list_1 = data["lg_node_list_1"]  # 边图的节点列表：[((7, 3), (3, 8), (3, 9),...)]
        lg_node_list_2 = data["lg_node_list_2"]
        lg_edge_index_mapping_1 = data["lg_edge_index_mapping_1"]  # 边图的边索引映射[((7, 3), (3, 9)),...]->[(0, 2),...]
        lg_edge_index_mapping_2 = data["lg_edge_index_mapping_2"]
        lg_features_1 = data["lg_features_1"]
        lg_features_2 = data["lg_features_2"]
        # print(lg_features_1.shape)
        # print(lg_features_2.shape)
        # print(edge_index_1.shape)
        # print(edge_index_1)
        # print(edge_index_2.shape)
        # print(edge_index_2)
        # print(lg_edge_index_mapping_1.shape)
        # print(lg_edge_index_mapping_1)
        # print(lg_edge_index_mapping_2.shape)
        # print(lg_edge_index_mapping_2)
        # print("---------------------------------")
        # time.sleep(3)
        
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        
        flag_1 = 0
        flag_2 = 0
        if len(lg_node_list_1) == 1: 
            flag_1 = 1
            lg_features_1 = torch.cat([lg_features_1, lg_features_1], dim=0)
        if len(lg_node_list_2) == 1: 
            flag_2 = 1
            lg_features_2 = torch.cat([lg_features_2, lg_features_2], dim=0)
        lg_abstract_features_1 = self.lg_convolutional(lg_edge_index_mapping_1, lg_features_1)
        lg_abstract_features_2 = self.lg_convolutional(lg_edge_index_mapping_2, lg_features_2)
        if flag_1: lg_abstract_features_1 = lg_abstract_features_1[0:1]
        if flag_2: lg_abstract_features_2 = lg_abstract_features_2[0:1]

        cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)  # n*m
        map_matrix = self.mapMatrix(abstract_features_1, abstract_features_2)  # n*m
        lg_map_matrix = self.lg_mapMatrix(lg_abstract_features_1, lg_abstract_features_2)  # en*em
        
        # print(map_matrix)
        # print(lg_map_matrix)
        
        node_alignment = self.node_alignment_with_edge(map_matrix, lg_map_matrix, lg_node_list_1, lg_node_list_2)  # n*m(0-1矩阵)
        # time.sleep(10)
        # map_matrix = self.transport_plan(map_matrix)

        # calculate ged using map_matrix
        m = torch.nn.Softmax(dim=1)
        # soft_matrix = map_matrix * cost_matrix
        soft_matrix = node_alignment * cost_matrix
        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2)
        score = torch.sigmoid(soft_matrix.sum() + bias_value)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item(), map_matrix

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