import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class LSTM(nn.Module):
    def __init__(self, input_size, seq_len, num_layer, hidden_size, dropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = LstmLayer(input_size, num_layer, hidden_size, dropout)
        self.data = None
        self.seq_len = seq_len

    def forward(self, x):
        if self.data is None:
            self.data = self.generate_data(x)
        x = self.lstm(self.data)
        return x

    def generate_data(self, x):
        # x shape: (n, feat_num)
        # data shape: (n, seq_len, feat_num)
        n = x.shape[0]
        feat_num = x.shape[1]
        feat = []
        for i in range(n):
            feat.append(x[max(0, i - self.seq_len):i + 1, :])
        # padding feat to make it have same length
        feat = pad_sequence(feat, True, 0)
        return feat


class LstmLayer(nn.Module):
    def __init__(self, input_size, num_layer, hidden_size, dropout):
        super(LstmLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, dropout=dropout,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out.reshape(-1, 1)
        out = self.fc(out)
        return out
