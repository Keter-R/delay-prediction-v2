import yaml
from codes.loader.data_loader import m_DataLoader
from codes.models.MLP import MLP
from codes.models.GCN import GCN
from codes.models.GCN import std_GCN

def load_model(config, data):
    data_feature = data['data_feature']
    data_length = data['data_length']
    temporal_adj = data['temporal_adj']
    knn_adj = data['knn_adj']

    models = dict()
    if 'std_gcn_temporal' in config and config['std_gcn_temporal']['enable']:
        print('Loading std GCN model...')
        models['std_gcn_temporal'] = std_GCN(data_feature, config['std_gcn_temporal']['layer_dim'],
                                             config['std_gcn_temporal']['layer_num'],
                                             temporal_adj, config['std_gcn_temporal']['dropout'])
    if 'gcn_temporal' in config and config['gcn_temporal']['enable']:
        print('Loading GCN model...')
        models['gcn_temporal'] = GCN(data_feature, config['gcn_temporal']['layer_dim'],
                                     temporal_adj, config['gcn_temporal']['dropout'],)
    if 'gcn_knn' in config and config['gcn_knn']['enable']:
        print('Loading GCN model...')
        models['gcn_knn'] = GCN(data_feature, config['gcn_knn']['layer_dim'],
                                knn_adj, config['gcn_knn']['dropout'])
    if 'mlp' in config and config['mlp']['enable']:
        print('Loading MLP model...')
        models['mlp'] = MLP(data_feature, data_length, config['mlp']['layer_dim'], config['mlp']['dropout'])
    return models


def load_data(seed, config, using_temporal=False, using_knn=False):
    dat = m_DataLoader(seed=seed, using_temporal=using_temporal, using_knn=using_knn,
                       split_ratio=config['split_ratio'], time_duration=config['time_dummy_duration'],
                       temporal_config=config['temporal_graph'], knn_config=config['knn_graph'])
    data = dict()
    data['x_train'] = dat.x_train
    data['y_train'] = dat.y_train
    data['x_val'] = dat.x_val
    data['y_val'] = dat.y_val
    data['data_module'] = dat.data_module
    data['data_length'] = dat.data_length
    data['data_feature'] = dat.data_feature
    data['np_ratio'] = dat.np_ratio
    data['knn_adj'] = dat.knn_adj
    data['temporal_adj'] = dat.temporal_adj
    return data
