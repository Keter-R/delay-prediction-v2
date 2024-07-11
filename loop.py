import os

import pytorch_lightning as pl

from codes import loader


def Task(seed, config):
    pl.seed_everything(seed=seed, workers=True)
    data = loader.load_data(seed, config['data'],
                            using_temporal=config['using_temporal_graph'],
                            using_knn=config['using_knn_graph'])
    models = loader.load_model(config['models'], data['data_feature'], data['data_length'])
    return dict(), dict()
