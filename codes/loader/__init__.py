import yaml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from codes.loader.data_loader import m_DataLoader
from codes.models.MLP import MLP
from codes.models.GCN import GCN, LS_GCN
from codes.models.GCN import std_GCN
from codes.models.LSTM import LSTM
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import SGDClassifier


def load_model(config, data, seed):
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
                                     temporal_adj, config['gcn_temporal']['dropout'], )
    if 'gcn_knn' in config and config['gcn_knn']['enable']:
        print('Loading GCN model...')
        models['gcn_knn'] = GCN(data_feature, config['gcn_knn']['layer_dim'],
                                knn_adj, config['gcn_knn']['dropout'])
    if 'mlp' in config and config['mlp']['enable']:
        print('Loading MLP model...')
        models['mlp'] = MLP(data_feature, data_length, config['mlp']['layer_dim'], config['mlp']['dropout'])
    if 'random_forest' in config and config['random_forest']['enable']:
        print('Loading Random Forest model...')
        models['random_forest'] = load_sci_kit_models(seed, config, 'random_forest')
    if 'balanced_random_forest' in config and config['balanced_random_forest']['enable']:
        print('Loading Balanced Random Forest model...')
        models['balanced_random_forest'] = load_sci_kit_models(seed, config, 'balanced_random_forest')
    if 'svm' in config and config['svm']['enable']:
        print('Loading SVM model...')
        models['svm'] = load_sci_kit_models(seed, config, 'svm')
    if 'regression' in config and config['regression']['enable']:
        print('Loading Regression model...')
        models['regression'] = load_sci_kit_models(seed, config, 'regression')
    if 'lstm' in config and config['lstm']['enable']:
        print('Loading LSTM model...')
        models['lstm'] = LSTM(data_feature, config['lstm']['seq_len'], config['lstm']['num_layer'],
                              config['lstm']['hidden_size'], config['lstm']['dropout'])

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


def load_sci_kit_models(seed, config, name):
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=config[name]['n_estimators'],
                                      max_features=config[name]['max_features'],
                                      criterion=config[name]['criterion'],
                                      class_weight=None if 'class_weight' not in config[name] else config[name][
                                          'class_weight'],
                                      random_state=seed, n_jobs=-1)
    if name == 'balanced_random_forest':
        return BalancedRandomForestClassifier(n_estimators=config[name]['n_estimators'],
                                              max_features=config[name]['max_features'],
                                              criterion=config[name]['criterion'],
                                              random_state=seed, n_jobs=-1)
    if name == 'svm':
        return SGDClassifier(loss='hinge', penalty='l2', eta0=0.01,
                             random_state=seed, max_iter=5, learning_rate='adaptive', class_weight='balanced')

    if name == 'regression':
        return SGDClassifier(loss='log_loss', penalty='l2', eta0=0.01,
                             random_state=seed, max_iter=100, learning_rate='adaptive', class_weight='balanced')