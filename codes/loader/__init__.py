import yaml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from codes.loader.data_loader import m_DataLoader
from codes.models.MLP import MLP
from codes.models.GCN import GCN, std_GCN
from codes.models.GCN import GTN
from codes.models.GAT import GAT
from codes.models.LSTM import LSTM
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def load_model(config, data, seed):
    data_feature = data['data_feature']
    data_length = data['data_length']
    temporal_adj = data['temporal_adj']
    knn_adj = data['knn_adj']

    models = dict()
    if 'gat' in config and config['gat']['enable']:
        print('Loading GAT model...')
        models['gat'] = GAT(data_feature, config['gat']['hidden_size'], temporal_adj,
                            config['gat']['heads'], config['gat']['dropout'])
    if 'gtn' in config and config['gtn']['enable']:
        print('Loading std gtn model...')
        models['gtn'] = GTN(data_feature, config['gtn']['layer_dim'], temporal_adj, config['gtn']['heads'], config['gtn']['dropout'])
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
    if 'random_forest_balanced' in config and config['random_forest_balanced']['enable']:
        print('Loading Balanced Random Forest model...')
        models['random_forest_balanced'] = load_sci_kit_models(seed, config, 'random_forest_balanced')
    if 'svm' in config and config['svm']['enable']:
        print('Loading SVM model...')
        models['svm'] = load_sci_kit_models(seed, config, 'svm')
    if 'svm_balanced' in config and config['svm_balanced']['enable']:
        print('Loading Balanced SVM model...')
        models['svm_balanced'] = load_sci_kit_models(seed, config, 'svm_balanced')
    if 'regression' in config and config['regression']['enable']:
        print('Loading Regression model...')
        models['regression'] = load_sci_kit_models(seed, config, 'regression')
    if 'regression_balanced' in config and config['regression_balanced']['enable']:
        print('Loading Balanced Regression model...')
        models['regression_balanced'] = load_sci_kit_models(seed, config, 'regression_balanced')
    if 'std_gcn' in config and config['std_gcn']['enable']:
        print('Loading std_gcn model...')
        models['std_gcn'] = std_GCN(data_feature, temporal_adj, config['std_gcn']['num_layer'],
                                    config['std_gcn']['hidden_size'], config['std_gcn']['dropout'])

    return models


def load_data(seed, config, using_temporal=False, using_knn=False, removed_features=None):
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
    data['data'] = dat.data
    data['val_index'] = dat.val_index
    data['edge_index'] = dat.edge_index
    return data


def load_sci_kit_models(seed, config, name):
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=config[name]['n_estimators'],
                                      max_features=config[name]['max_features'],
                                      criterion=config[name]['criterion'],
                                      class_weight=None if 'class_weight' not in config[name] else config[name][
                                          'class_weight'],
                                      random_state=seed, n_jobs=-1)
    if name == 'random_forest_balanced':
        return BalancedRandomForestClassifier(n_estimators=config[name]['n_estimators'],
                                              max_features=config[name]['max_features'],
                                              criterion=config[name]['criterion'],
                                              random_state=seed, n_jobs=-1)
    if name == 'svm':
        if config[name]['sgd']:
            return SGDClassifier(loss='hinge', penalty='l2', eta0=0.01,
                                 random_state=seed, max_iter=5, learning_rate='adaptive')
        else:
            return SVC(kernel='rbf', class_weight=None, probability=True)

    if name == 'svm_balanced':
        if config[name]['sgd']:
            return SGDClassifier(loss='hinge', penalty='l2', eta0=0.01,
                                 random_state=seed, max_iter=5, learning_rate='adaptive', class_weight='balanced')
        else:
            return SVC(kernel='rbf', class_weight='balanced', probability=True)
    if name == 'regression':
        if config[name]['sgd']:
            return SGDClassifier(loss='log_loss', penalty='l2', eta0=0.01,
                                 random_state=seed, max_iter=100, learning_rate='adaptive')
        else:
            return LogisticRegression(random_state=seed)
    if name == 'regression_balanced':
        if config[name]['sgd']:
            return SGDClassifier(loss='log_loss', penalty='l2', eta0=0.01,
                                 random_state=seed, max_iter=100, learning_rate='adaptive', class_weight='balanced')
        else:
            return LogisticRegression(random_state=seed, class_weight='balanced')
