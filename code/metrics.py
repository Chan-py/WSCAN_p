import math
import networkx as nx
import numpy as np

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score


########### ARI ###############
def compute_ARI(clusters, ground_truth):
    """
    Compute Adjusted Rand Index (ARI) given:
      - clusters: dict mapping cluster_id -> iterable of nodes
      - ground_truth: dict mapping node -> true_label (can be int or str)

    Returns:
      - ARI float in [-1,1]
    """
    # 1) Build predicted label mapping
    y_pred_map = {}
    for cid, members in clusters.items():
        for u in members:
            y_pred_map[u] = cid
    # 2) Prepare y_true, y_pred lists sorted by node ID
    nodes = sorted(ground_truth.keys())
    y_true = [ground_truth[u] for u in nodes]
    # assign any node not in a cluster to a special label (-1)
    y_pred = [y_pred_map.get(u, -1) for u in nodes]
    # 3) Compute ARI
    return adjusted_rand_score(y_true, y_pred)

########### NMI
def compute_NMI(clusters, ground_truth, average_method='arithmetic'):
    """
    Compute Normalized Mutual Information (NMI).
      - clusters: dict {cluster_id -> iterable_of_nodes}
      - ground_truth: dict {node -> true_label}
      - average_method: {'min', 'geometric', 'arithmetic', 'max'} (sklearn 옵션)
    """
    y_pred_map = {}
    for cid, members in clusters.items():
        for u in members:
            y_pred_map[u] = cid

    nodes = sorted(ground_truth.keys())
    y_true = [ground_truth[u] for u in nodes]
    y_pred = [y_pred_map.get(u, -1) for u in nodes]

    return normalized_mutual_info_score(
        y_true, y_pred, average_method=average_method
    )