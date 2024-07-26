import math

import dgl
import torch
import torch.nn as nn
import torch_geometric.utils
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCNConv
import torch_geometric.nn.models as tgmodels


class std_GCN(nn.Module):
    def __init__(self, in_features, layer_dim, layer_num, adj, dropout=0.):
        super(std_GCN, self).__init__()
        self.in_features = in_features
        adj = torch.tensor(adj, dtype=torch.float32).to('cuda')
        D = torch.diag(torch.sum(adj, dim=1))
        D_inv = torch.inverse(D)
        A_hat = torch.matmul(torch.matmul(D_inv, adj), D_inv)
        self.A = adj.to('cuda')
        self.edge_index, self.edge_weight = torch_geometric.utils.dense_to_sparse(self.A)
        self.gcn = tgmodels.GCN(in_features, layer_dim, layer_num, 1, dropout)

    def forward(self, feat):
        # normalize
        feat = feat / feat.sum(dim=1, keepdim=True)
        feat = self.gcn(feat, self.edge_index, self.edge_weight)
        return feat


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
        # D_inv equals -1/2 power of D
        self.A = torch.tensor(self.A, dtype=torch.float32).to('cuda')
        D = torch.diag(torch.sum(self.A, dim=1))
        D_inv = torch.inverse(D)
        D_inv = torch.sqrt(D_inv)
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
            self.in_process.add_module('in_norm', nn.BatchNorm1d(c_in))
        if activation is not None:
            self.out_process.add_module('out_activation', activation)
        if dropout is not None and dropout > 0:
            self.out_process.add_module('out_dropout', nn.Dropout(dropout))
        self.gcn = GCNConv(c_in, c_out, cached=True, add_self_loops=False, normalize=False)

    def forward(self, node_feats):
        node_feats = self.in_process(node_feats)
        node_feats = self.gcn(node_feats, self.edge_index, self.edge_weight)
        node_feats = self.out_process(node_feats)
        return node_feats


class LS_GCN(nn.Module):
    def __init__(self, in_features, dim_list, adj, dropout=0.0, seq_len=4):
        super(LS_GCN, self).__init__()
        self.in_features = in_features
        self.hidden_dim = dim_list
        self.dropout = dropout
        self.A = adj
        self.lstm_out = 16
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=self.lstm_out, num_layers=1,
                            batch_first=True, dropout=dropout)
        self.gcn = self.generate_gcn()
        self.data = None
        self.x = None

    def generate_data(self, x):
        feat = []
        for i in range(x.shape[0]):
            feat.append(x[max(0, i - self.seq_len - 1):i, :])
        # padding feat to make it have same length
        feat = pad_sequence(feat, True, 0)
        return feat

    def forward(self, feat):
        # normalize
        feat = feat / feat.sum(dim=1, keepdim=True)
        if self.x is None:
            self.x = feat.clone()
        if self.data is None:
            self.data = self.generate_data(feat)
        feat, _ = self.lstm(self.data)
        feat = feat[:, -1, :]
        # concat [x feat]
        feat = torch.cat([self.x, feat], dim=1)
        for layer in self.gcn:
            feat = layer(feat)
        return feat

    def generate_gcn(self):
        # D_inv equals -1/2 power of D
        self.A = torch.tensor(self.A, dtype=torch.float32).to('cuda')
        D = torch.diag(torch.sum(self.A, dim=1))
        D_inv = torch.inverse(D)
        D_inv = torch.sqrt(D_inv)
        A_hat = torch.matmul(torch.matmul(D_inv, self.A), D_inv)
        self.A = A_hat.to('cuda')
        layers = []
        for i in range(len(self.hidden_dim) - 1):
            if i == 0:
                layers.append(
                    GCNLayer(self.lstm_out + self.in_features, self.hidden_dim[i], self.A, self.dropout, nn.LeakyReLU(), False))
            else:
                layers.append(
                    GCNLayer(self.hidden_dim[i - 1], self.hidden_dim[i], self.A, self.dropout, nn.LeakyReLU()))
        layers.append(GCNLayer(self.hidden_dim[-2], self.hidden_dim[-1], self.A, None, None))
        return nn.ModuleList(layers)
