import os
import torch
import yaml
from codes import loader
from loop import Task


# delete cache files: temporal_graph.npy, knn_graph.npy
# if os.path.exists('./temporal_adj.npy'):
#     os.remove('./temporal_adj.npy')
# if os.path.exists('./knn_adj.npy'):
#     os.remove('./knn_adj.npy')


config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
# random_seed = [2, 3, 5, 7, 11, 13, 17, 19, 23, 998244353]
random_seed = [1, 2]
torch.set_float32_matmul_precision('high')

for seed in random_seed:
    metrics, curves = Task(seed, config)


# delete cache files: temporal_graph.npy, knn_graph.npy
# os.remove('./temporal_adj.npy')
# os.remove('./knn_adj.npy')
