| Performance   | AUROC           | Accuracy        | GMean           | F1              | Sensitivity     | Specificity     |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| GCN(temporal) | __0.812±0.009__ | 0.727±0.015     | __0.737±0.010__ | 0.312±0.011     | __0.750±0.022__ | 0.725±0.017     |
| MLP           | 0.802±0.011     | __0.792±0.052__ | 0.715±0.036     | __0.346±0.036__ | 0.645±0.100     | __0.805±0.065__ |

GCN(temporal): time_interval = 45, self_weight=20, gcn_layer=[512, 256, 1], lr = 0.003, lr_decay = 0.5 per 20 epochs, weight_decay = 0.001, dropout = 0.1, loss_weight = 1.5 * np_ratio, epoch = 200