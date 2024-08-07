import math

import dgl
import torch
import torch.nn as nn
from codes.models.gat_tp import models
from torch_geometric.nn.models import GAT as PYG_GAT
import torch_geometric.utils

class GAT(nn.Module):
    def __init__(self, in_features, hidden_size, adj, heads=3, dropout=None):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.dropout = dropout
        self.A = torch.tensor(adj, dtype=torch.float32).to('cuda')
        self.edge_index, self.edge_weight = torch_geometric.utils.dense_to_sparse(self.A)
        self.gat_out = 1
        # set values in A to 1 if not 0
        # self.A[self.A != 0] = 1
        self.gat = models.GAT(in_features, hidden_size, self.gat_out, dropout, 0.2, heads)
        # self.gat = PYG_GAT(in_features, hidden_size, 5, 1, dropout, v2=True, jk='lstm')
        # self.fc = nn.Linear(self.gat_out, 1)

    def forward(self, feat):
        # normalize

        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat, dtype=torch.float32).to('cuda')
        # feat = feat / feat.sum(dim=1, keepdim=True)
        if isinstance(self.gat, PYG_GAT):
            feat = self.gat(feat, self.edge_index, self.edge_weight)
        else:
            feat = self.gat(feat, self.A)
        # print(feat)
        # feat = self.fc(feat)
        # print(feat)
        # print(feat)
        # exit(13)
        # print(feat.shape)
        # exit(133113)
        return feat
