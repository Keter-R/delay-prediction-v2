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
    # for model_name, metric in metrics_list[0].items():
    #     for key, value in metric.items():
    #         best_value = max([metrics[model_name][key] for metrics in metrics_list])
    #         writer.add_text(model_name + '_' + key + '_max', best_value)
    # add metrics mean and std
    metrics_std, metrics_mean = calculate_metrics_std_and_mean(metrics_list)
    for model_name, metric in metrics_mean.items():
        for key, value in metric.items():
            # makes val a string that in form of 'value±std' with .3f precision
            val = str(round(value, 3)) + '±' + str(round(metrics_std[model_name][key], 3))
            writer.add_text(model_name + '_' + key, val)
    writer.flush()
    writer.close()
    return metrics_std, metrics_mean


def out_to_markdown_form(metrics_std, metrics_mean):
    data_cols = ['Accuracy', 'AUC', 'F1_score', 'GMean', 'Sensitivity', 'Specificity']
    data_max = dict()
    markdown_source = '| Model | ' + ' | '.join(data_cols) + ' |\n'
    markdown_source += '| --- | --- | --- | --- | --- | --- | --- |\n'

    for model_name in metrics_mean.keys():
        for key in data_cols:
            if key not in data_max.keys() or metrics_mean[model_name][key] > data_max[key]:
                data_max[key] = metrics_mean[model_name][key]
    for k, v in data_max.items():
        data_max[k] = str(round(v, 3))
        while len(data_max[k].split('.')[1]) < 3:
            data_max[k] += '0'
    for model_name in metrics_mean.keys():
        markdown_source += '| ' + model_name + ' | '
        for key in data_cols:
            _mean = str(round(metrics_mean[model_name][key], 3))
            _std = str(round(metrics_std[model_name][key], 3))
            # makes val a string that in form of 'value±std' with .3f precision, add 0 suffix to make it 3 digits
            while len(_mean.split('.')[1]) < 3:
                _mean += '0'
            while len(_std.split('.')[1]) < 3:
                _std += '0'
            if _mean == data_max[key]:
                _mean = '**' + _mean + '**'
            val = _mean + '±' + _std
            markdown_source += val + ' | '
        markdown_source += '\n'
    return markdown_source
