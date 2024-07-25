import os

import numpy as np
import pandas as pd
import torch
from numpy import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklego.preprocessing import RepeatingBasisFunction
from codes.loader import graph_utils


class m_DataLoader:
    def __init__(self,
                 seed,
                 time_duration=0,
                 using_temporal=False, using_knn=False,
                 temporal_config=None, knn_config=None,
                 split_ratio=0.8,
                 data_path='./data/ttc-streetcar-delay-data-2023-pure.xlsx'):
        self.seed = seed
        random.seed(seed)
        self.data_path = data_path
        self.time_duration = time_duration
        self.using_temporal = using_temporal
        self.using_knn = using_knn
        self.temporal_config = temporal_config
        self.knn_config = knn_config
        self.split_ratio = split_ratio
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.data = None
        self.train_index = None
        self.val_index = None
        self.data_module = None
        self.raw_data = None
        self.data_length = None
        self.data_feature = None
        self.knn_adj = None
        self.temporal_adj = None
        self.np_ratio = None
        self.load()

    def load(self):
        dat = pd.read_excel(self.data_path)
        dat['Delay'] = dat['Delay'].apply(lambda x: 1 if x > 30 else 0)
        self.raw_data = dat
        self.data = self.process_raw()
        self.data_length = len(dat)
        self.sample()
        self.generate_data_module()
        self.generate_data()

    def generate_data(self):
        self.x_train = self.data.loc[self.train_index].drop(columns=['Delay'])
        self.y_train = self.data.loc[self.train_index]['Delay']
        self.x_val = self.data.loc[self.val_index].drop(columns=['Delay'])
        self.y_val = self.data.loc[self.val_index]['Delay']
        # drop column names and convert to numpy array
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_val = np.array(self.x_val)
        self.y_val = np.array(self.y_val)

    def generate_data_module(self):
        self.data_module = DataModule(self.data, self.train_index, self.val_index,
                                      self.using_temporal, self.using_knn,
                                      self.temporal_config, self.knn_config,
                                      self.raw_data['Date'])
        self.temporal_adj = self.data_module.temporal_adj
        self.knn_adj = self.data_module.knn_adj

    def process_raw(self):
        dat = self.raw_data.copy()
        dat['Date'] = pd.to_datetime(dat['Date'])
        dat.insert(column='Time', value=dat['Date'].dt.time, loc=dat.columns.get_loc('Date'))
        if self.time_duration > 0:
            dat['Time'] = (dat['Date'].dt.hour * 60 + dat['Date'].dt.minute) // self.time_duration
        else:
            dat['Time'] = dat['Time'].apply(lambda x: 'Morning' if 6 <= x.hour < 12
                                            else 'Afternoon' if 12 <= x.hour < 18
                                            else 'Evening')
        # dat = dat.rename(columns={'Date': 'Month'})
        # dat['Month'] = dat['Month'].dt.month
        # dat = pd.get_dummies(dat, columns=['Route', 'Time', 'Day', 'Month', 'Incident'])
        flag = True

        if flag:
            N_PERIODS = 12
            dat['Time'] = dat['Date'].dt.hour * 60 + dat['Date'].dt.minute
            dat['Date'] = dat['Date'].dt.dayofyear
            rbf_date = RepeatingBasisFunction(
                n_periods=N_PERIODS,
                remainder="drop",
                column="Date",
                input_range=(1, 365)
            )
            date_cols = rbf_date.fit_transform(dat)
            rbf_time = RepeatingBasisFunction(
                n_periods=N_PERIODS,
                remainder="drop",
                column="Time",
                input_range=(0, 24 * 60)
            )
            time_cols = rbf_time.fit_transform(dat)
            dat = dat.drop(columns=['Date', 'Day', 'Time'])
            date_cols = pd.DataFrame(date_cols, columns=[f'Date_{i}' for i in range(N_PERIODS)])
            time_cols = pd.DataFrame(time_cols, columns=[f'Time_{i}' for i in range(N_PERIODS)])
            dat = pd.concat([dat, date_cols], axis=1)
            dat = pd.concat([dat, time_cols], axis=1)
            dat = pd.get_dummies(dat, columns=['Route', 'Incident'])
            dat = dat.astype(float)
            print(dat.columns)
        self.data_feature = len(dat.columns) - 1
        return dat

    def sample(self):
        positive_indexes = [i for i in range(0, len(self.data)) if self.data['Delay'][i] == 1]
        negative_indexes = [i for i in range(0, len(self.data)) if self.data['Delay'][i] == 0]
        random.shuffle(positive_indexes)
        random.shuffle(negative_indexes)
        train_positive = positive_indexes[:int(len(positive_indexes) * self.split_ratio)]
        train_negative = negative_indexes[:int(len(negative_indexes) * self.split_ratio)]
        val_positive = positive_indexes[int(len(positive_indexes) * self.split_ratio):]
        val_negative = negative_indexes[int(len(negative_indexes) * self.split_ratio):]
        print('train NP ratio: ', len(train_negative) / len(train_positive))
        print('val NP ratio: ', len(val_negative) / len(val_positive))
        self.np_ratio = len(train_negative) / len(train_positive)
        self.train_index = train_positive
        self.train_index.extend(train_negative)
        self.val_index = val_positive
        self.val_index.extend(val_negative)
        print('Negative Positive Ratio: ', self.np_ratio)

    def data_module(self):
        return self.data_module

    def train_data(self):
        return self.x_train, self.y_train

    def val_data(self):
        return self.x_val, self.y_val


class DataModule(pl.LightningDataModule):
    def __init__(self, data, train_index, val_index, using_temporal, using_knn, temporal_config, knn_config, time_col):
        super(DataModule, self).__init__()
        self.data = data
        self.train_index = train_index
        self.val_index = val_index
        self.using_temporal = using_temporal
        self.using_knn = using_knn
        self.temporal_config = temporal_config
        self.knn_config = knn_config
        self.time_col = time_col
        self.train_dataset = None
        self.val_dataset = None
        self.temporal_adj = None
        self.knn_adj = None
        self.generate_data()

    def generate_data(self):
        n = self.data.shape[0]
        feat_num = self.data.shape[1] - 1
        X = self.data.drop(columns=['Delay'])
        Y = self.data['Delay']
        feat = np.array(X)
        y_train = np.concatenate([self.train_index, Y.iloc[self.train_index]]).T
        y_val = np.concatenate([self.val_index, Y.iloc[self.val_index]]).T
        temporal_adj, knn_adj = self.generate_adj()
        # squeeze feat, temporal_adj, knn_adj together
        X = feat.flatten()
        assert X.shape[0] == n * feat_num
        assert y_train.shape[0] == len(self.train_index * 2)
        assert y_val.shape[0] == len(self.val_index * 2)
        X = torch.tensor(X, dtype=torch.float32).reshape(1, -1)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(1, -1)
        y_val = torch.tensor(y_val, dtype=torch.float32).reshape(1, -1)
        self.train_dataset = torch.utils.data.TensorDataset(X, y_train)
        self.val_dataset = torch.utils.data.TensorDataset(X, y_val)

    def generate_adj(self):
        n = self.data.shape[0]
        temporal_adj = np.zeros((n, n))
        knn_adj = np.zeros((n, n))
        if self.using_temporal and self.temporal_config is not None:
            if self.temporal_config['using_cache'] and os.path.exists('./temporal_adj.npy'):
                temporal_adj = np.load('./temporal_adj.npy')
            else:
                temporal_adj = graph_utils.generate_temporal_graph(self.time_col,
                                                                   self.temporal_config['time_interval'],
                                                                   self.temporal_config['self_weight'],
                                                                   self.temporal_config['weighted_edge'])
                np.save('./temporal_adj.npy', temporal_adj)
        if self.using_knn and self.knn_config is not None:
            if self.knn_config['using_cache'] and os.path.exists('./knn_adj.npy'):
                knn_adj = np.load('./knn_adj.npy')
            else:
                knn_adj = graph_utils.generate_knn_graph(self.data.drop(columns=['Delay']),
                                                         self.knn_config['k'],
                                                         self.knn_config['self_weight'],
                                                         self.knn_config['weighted_edge'])
                np.save('./knn_adj.npy', knn_adj)
        self.temporal_adj = temporal_adj
        self.knn_adj = knn_adj
        return temporal_adj, knn_adj

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)
