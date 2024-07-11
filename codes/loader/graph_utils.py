import numpy as np
from joblib import Parallel, delayed

threads_count = -1


def generate_temporal_graph(time_col, intervals, self_weight=1, weighted_edge=False):
    # convert pd_time to seconds to minutes
    time_col = np.array([int(i.timestamp()) for i in time_col]) // 60
    results = Parallel(n_jobs=threads_count)(
        delayed(temporal_parallel)(i, len(time_col), time_col, intervals, weighted_edge) for i in range(len(time_col))
    )
    results = np.array(results)
    results += self_weight * np.eye(len(time_col))
    return results


def generate_knn_graph(data, k, self_weight=1, weighted_edge=False):
    data = np.array(data.astype(float))
    results = Parallel(n_jobs=threads_count)(
        delayed(knn_parallel)(i, len(data), data, k, weighted_edge) for i in range(len(data))
    )
    results = np.array(results)
    results += self_weight * np.eye(len(data))
    return results


def temporal_parallel(x, n, time_col, intervals, weighted_edge=False):
    edge = np.zeros(n)
    for i in range(n):
        if x != i and abs((time_col[x] - time_col[i])) < intervals:
            if weighted_edge:
                edge[i] = 1 - abs((time_col[x] - time_col[i])) / intervals
            else:
                edge[i] = 1
    return edge


def knn_parallel(x, n, data, k, weighted_edge=False):
    edge = np.zeros(n)
    dist = np.linalg.norm(data - data[x], axis=1)
    idx = np.argsort(dist)
    for i in range(k):
        if x != idx[i]:
            if weighted_edge:
                # avoid divide by zero
                edge[idx[i]] = 1 - dist[idx[i]] / (dist[idx[k]] + 1e-6)
            else:
                edge[idx[i]] = 1
    return edge
