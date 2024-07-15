import math

import dgl
import torch
import torch.nn as nn
import torch_geometric.utils
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_features, dim_list, adj, dropout=None):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.hidden_dim = dim_list
        self.dropout = dropout
        self.A = adj
        self.gcn = self.generate_gcn()

    def forward(self, feat):
        # normalize
        feat = feat / feat.sum(dim=1, keepdim=True)
        for layer in self.gcn:
            feat = layer(feat)
        return feat

    def generate_gcn(self):
        self.A = torch.tensor(self.A, dtype=torch.float32)
        D = torch.diag(torch.sum(self.A, dim=1))
        D_inv = torch.inverse(D)
        A_hat = torch.matmul(torch.matmul(D_inv, self.A), D_inv)
        self.A = A_hat.to('cuda')
        layers = []
        # for i in range(len(self.hidden_dim)):
        #     gc = []
        #     if i == 0:
        #         gc.append(nn.BatchNorm1d(self.in_features))
        #         gc.append(nn.Linear(self.in_features, self.hidden_dim[i]))
        #     else:
        #         gc.append(nn.BatchNorm1d(self.hidden_dim[i - 1]))
        #         gc.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
        #     gc.append(nn.ReLU())
        #     if self.dropout is not None and self.dropout > 0 and i != len(self.hidden_dim) - 1:
        #         gc.append(nn.Dropout(self.dropout))
        #     layers.append(nn.Sequential(*gc))
        for i in range(len(self.hidden_dim) - 1):
            if i == 0:
                layers.append(
                    GCNLayer(self.in_features, self.hidden_dim[i], self.A, self.dropout, nn.LeakyReLU(), False))
            else:
                layers.append(
                    GCNLayer(self.hidden_dim[i - 1], self.hidden_dim[i], self.A, self.dropout, nn.LeakyReLU()))
        layers.append(GCNLayer(self.hidden_dim[-2], self.hidden_dim[-1], self.A, None, None))
        return nn.ModuleList(layers)


class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out, adj_matrix, dropout=None, activation=None, norm=False):
        super().__init__()
        self.adj_matrix = adj_matrix.clone()
        # convert adj_matrix to 2 SparseTensor: index and weight
        self.edge_index, self.edge_weight = torch_geometric.utils.dense_to_sparse(self.adj_matrix)
        self.in_process = nn.Sequential()
        self.out_process = nn.Sequential()
        if norm:
            self.in_process.add_module('in_norm', nn.LayerNorm(c_in))
        if activation is not None:
            self.out_process.add_module('out_activation', activation)
        if dropout is not None and dropout > 0:
            self.out_process.add_module('out_dropout', nn.Dropout(dropout))
        self.gcn = GCNConv(c_in, c_out, cached=True, add_self_loops=False, normalize=True)

    def forward(self, node_feats):
        node_feats = self.in_process(node_feats)
        node_feats = self.gcn(node_feats, self.edge_index, self.edge_weight)
        node_feats = self.out_process(node_feats)
        return node_feats

# class GCNLayer(nn.Module):
#     """GCN层"""
#
#     def __init__(self, input_features, output_features, adj, bias=True):
#         super(GCNLayer, self).__init__()
#         self.input_features = input_features
#         self.output_features = output_features
#         self.adj = adj
#         self.weights = nn.Parameter(torch.FloatTensor(input_features, output_features))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(output_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         """初始化参数"""
#         std = 1. / math.sqrt(self.weights.size(1))
#         self.weights.data.uniform_(-std, std)
#         if self.bias is not None:
#             self.bias.data.uniform_(-std, std)
#
#     def forward(self, x):
#         support = torch.mm(x, self.weights)
#         output = torch.mm(self.adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         return output
