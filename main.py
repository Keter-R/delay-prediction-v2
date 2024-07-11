import os

import yaml
from codes import loader
from loop import Task
config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)


random_seed = [2, 3, 5, 7, 11, 13, 17, 19, 23, 998244353]

for seed in random_seed:
    metrics, curves = Task(seed, config)


# delete cache files: temporal_graph.npy, knn_graph.npy
os.remove('./temporal_adj.npy')
os.remove('./knn_adj.npy')
