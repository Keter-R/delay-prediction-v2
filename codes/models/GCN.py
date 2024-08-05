import math

import dgl
import torch
import torch.nn as nn
import torch_geometric.utils
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCNConv, TransformerConv
import torch_geometric.nn.models as tgmodels


class GTN(nn.Module):
    def __init__(self, in_features, layer_dim, adj,heads=5, dropout=0.):
        super(GTN, self).__init__()
        self.in_features = in_features
        self.layer_dim = layer_dim
        self.dropout = dropout
        self.heads = heads
        self.A = torch.tensor(adj, dtype=torch.float32).to('cuda')
        # set A[i,i] = 0
        self.A = self.A - torch.diag(self.A.diagonal())
        self.edge_index, self.edge_weight = torch_geometric.utils.dense_to_sparse(self.A)
        self.layers = self.generate_layers()

    def forward(self, feat):
        # normalize
        feat = feat / feat.sum(dim=1, keepdim=True)
        for layer in self.layers:
            feat = layer(feat)
        return feat

    def generate_layers(self):
        layers_dims = [self.in_features] + self.layer_dim
        layers = []
        for i in range(len(layers_dims) - 1):
            layers.append(GTConv(layers_dims[i], layers_dims[i + 1], self.edge_index, self.heads, self.dropout))
        return nn.ModuleList(layers)


class GTConv(nn.Module):
    def __init__(self, in_features, out_features, edge_index, heads=5, dropout=0.):
        super(GTConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_index = edge_index
        if out_features != 1:
            self.conv = TransformerConv(-1, out_features, heads=heads, dropout=dropout)
        else:
            # using MLP to replace the last layer
            in_features *= heads
            self.conv = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, out_features)
            )

    def forward(self, feat):
        if self.conv.__class__.__name__ == 'TransformerConv':
            feat = self.conv(feat, self.edge_index)
        else:
            feat = self.conv(feat)
        print(feat.shape)
        # exit(133)
        return feat


class GCN(nn.Module):
    def __init__(self, in_features, dim_list, adj, dropout=None):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.hidden_dim = dim_list
        self.dropout = dropout
        self.A = adj
        self.gcn = self.generate_gcn()

    def forward(self, feat, edge_index=None):
        # normalize

        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat, dtype=torch.float32).to('cuda')
        feat = feat / feat.sum(dim=1, keepdim=True)
        for layer in self.gcn:
            if edge_index is not None:
                feat = layer(feat, edge_index)
            else:
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

    def forward(self, node_feats, edge_index=None):
        if edge_index is not None:
            self.edge_index = edge_index
        node_feats = self.in_process(node_feats)
        node_feats = self.gcn(node_feats, self.edge_index, self.edge_weight)
        node_feats = self.out_process(node_feats)
        return node_feats


class std_GCN(nn.Module):
    def __init__(self, in_features, adj, num_layer, hidden_size, dropout=0.0):
        super(std_GCN, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.A = torch.tensor(adj, dtype=torch.float32).to('cuda')
        self.edge_index, self.edge_weight = torch_geometric.utils.dense_to_sparse(self.A)
        self.gcn = torch_geometric.nn.models.GCN(in_features, hidden_size, num_layer, 1, dropout)

    def forward(self, feat):
        # normalize
        feat = feat / feat.sum(dim=1, keepdim=True)
        feat = self.gcn(feat, self.edge_index, self.edge_weight)
        return feat


