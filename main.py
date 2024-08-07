import os
import torch
import torchmetrics
import yaml
from codes import loader
from loop import Task
from analysis_utils import tensorboard_draw_metrics, out_to_markdown_form

# delete cache files: temporal_graph.npy, knn_graph.npy
if os.path.exists('./temporal_adj.npy'):
    os.remove('./temporal_adj.npy')
if os.path.exists('./knn_adj.npy'):
    os.remove('./knn_adj.npy')


config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
random_seed = [2, 3, 5, 7, 11, 13, 17, 19, 23, 998244353]
# random_seed = [2, 97, 997, 9973, 99991, 999983, 9999991, 99999989, 999999937, 9999999967]
random_seed = [3]
torch.set_float32_matmul_precision('high')
metrics_list = []
feat_cols = ['Route_301', 'Route_304', 'Route_306', 'Route_310',
             'Route_501', 'Route_503', 'Route_504', 'Route_505', 'Route_506',
             'Route_507', 'Route_508', 'Route_509', 'Route_510', 'Route_511',
             'Route_512', 'Time_0', 'Time_1', 'Time_2', 'Day_Friday', 'Day_Monday',
             'Day_Saturday', 'Day_Sunday', 'Day_Thursday', 'Day_Tuesday',
             'Day_Wednesday', 'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5',
             'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11',
             'Month_12', 'Incident_Cleaning - Unsanitary',
             'Incident_Collision - TTC Involved', 'Incident_Diversion',
             'Incident_Emergency Services', 'Incident_General Delay',
             'Incident_Held By', 'Incident_Investigation',
             'Incident_Late Entering Service', 'Incident_Mechanical',
             'Incident_Operations', 'Incident_Overhead', 'Incident_Rail/Switches',
             'Incident_Security', 'Incident_Utilized Off Route']
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
markdown_res = out_to_markdown_form(mean, std)
# write to file
with open('result_metrics.md', 'w', encoding='utf-8') as f:
    f.write(markdown_res)
# delete cache files: temporal_graph.npy, knn_graph.npy
# os.remove('./temporal_adj.npy')
# os.remove('./knn_adj.npy')
