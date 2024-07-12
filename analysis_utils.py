import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def calculate_metrics_std_and_mean(metrics_list):
    metrics_std = dict()
    metrics_mean = dict()
    for model_name in metrics_list[0].keys():
        metrics_std[model_name] = dict()
        metrics_mean[model_name] = dict()
        for key in metrics_list[0][model_name].keys():
            metrics = [metric[model_name][key] for metric in metrics_list]
            metrics_std[model_name][key] = np.std(metrics)
            metrics_mean[model_name][key] = np.mean(metrics)
    return metrics_std, metrics_mean


def tensorboard_draw_metrics(metrics_list):
    writer = SummaryWriter(log_dir="./lightning_logs/conclusion")
    # add metrics per round by different seeds
    for i, metrics in enumerate(metrics_list):
        for model_name, metric in metrics.items():
            for key, value in metric.items():
                writer.add_scalar('__' + model_name + '_' + key, value)
    # add metrics best performance in all seeds rounds
    for model_name, metric in metrics_list[0].items():
        for key, value in metric.items():
            best_value = max([metrics[model_name][key] for metrics in metrics_list])
            writer.add_text(model_name + '_' + key + '_max', best_value)
    # add metrics mean and std
    metrics_std, metrics_mean = calculate_metrics_std_and_mean(metrics_list)
    for model_name, metric in metrics_mean.items():
        for key, value in metric.items():
            # makes val a string that in form of 'value±std' with .3f precision
            val = str(round(value, 3)) + '±' + str(round(metrics_std[model_name][key], 3))
            writer.add_text(model_name + '_' + key, val)
    writer.flush()
    writer.close()
