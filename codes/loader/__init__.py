import yaml
from codes.loader.data_loader import m_DataLoader


def load_model(config, data_feature, data_length):
    models = {}
    if 'gcn' in config and config['gcn']['enable']:
        print('Loading GCN model...')
        models['gcn'] = 'GCN model loaded'
    if 'mlp' in config and config['mlp']['enable']:
        print('Loading MLP model...')
        models['mlp'] = 'MLP model loaded'
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
    return data
