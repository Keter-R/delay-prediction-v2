import os
import torch
import torchmetrics
import yaml
from codes import loader
from loop import Task
from analysis_utils import tensorboard_draw_metrics


# delete cache files: temporal_graph.npy, knn_graph.npy
if os.path.exists('./temporal_adj.npy'):
    os.remove('./temporal_adj.npy')
if os.path.exists('./knn_adj.npy'):
    os.remove('./knn_adj.npy')


config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
random_seed = [2, 3, 5, 7, 11, 13, 17, 19, 23, 998244353]
# random_seed = [2, 97, 997, 9973, 99991, 999983, 9999991, 99999989, 999999937, 9999999967]
# random_seed = [2]
torch.set_float32_matmul_precision('high')
metrics_list = []

# for i in range(1, 46):
#     config['data']['temporal_graph']['self_weight'] = i
#     if os.path.exists('./temporal_adj.npy'):
#         os.remove('./temporal_adj.npy')
for seed in random_seed:
    metrics, curves = Task(seed, config)
    metrics_list.append(metrics)

mean, std = tensorboard_draw_metrics(metrics_list)
print(mean)
print(std)
# delete cache files: temporal_graph.npy, knn_graph.npy
# os.remove('./temporal_adj.npy')
# os.remove('./knn_adj.npy')
