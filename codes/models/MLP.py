from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, data_feature, data_length, hidden_list, dropout=0):
        super(MLP, self).__init__()
        self.data_feature = data_feature
        self.data_length = data_length
        self.hidden_list = hidden_list
        self.dropout = dropout
        self.mlp = self.generate_mlp()

    def forward(self, feat):
        feat = self.mlp(feat)
        return feat

    def generate_mlp(self):
        layers = []
        for i in range(len(self.hidden_list)):
            if i == 0:
                layers.append(nn.LayerNorm(self.data_feature))
                layers.append(nn.Linear(self.data_feature, self.hidden_list[i]))
            else:
                layers.append(nn.LayerNorm(self.hidden_list[i - 1]))
                layers.append(nn.Linear(self.hidden_list[i - 1], self.hidden_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.LayerNorm(self.hidden_list[-1]))
        layers.append(nn.Linear(self.hidden_list[-1], 1))
        return nn.Sequential(*layers)
