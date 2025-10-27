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


import networkx as nx

def compute_modularity(G, clusters):
    """
    Weighted modularity for an undirected graph G and a hard partition 'clusters'.
    clusters: dict {cluster_id -> iterable_of_nodes}
    Formula: Q = sum_C [ w_in(C)/m - (vol(C)/(2m))^2 ]
      - w_in(C): sum of weights of edges with both ends in C (self-loops counted once)
      - vol(C): sum of node strengths in C (strength = weighted degree)
      - m: total edge weight (each undirected edge counted once)
    """
    # total edge weight (each undirected edge counted once)
    m = G.size(weight='weight')
    if m == 0:
        return 0.0  # no edges → modularity defined as 0

    # node strength (weighted degree)
    strength = dict(G.degree(weight='weight'))
    two_m = 2.0 * m

    Q = 0.0
    for cid, nodes in clusters.items():
        S = set(nodes)
        if not S:
            continue

        # internal weight (counted once, self-loops included once)
        # nx.subgraph(...).size(weight='weight') sums each undirected edge weight once
        w_in = G.subgraph(S).size(weight='weight')

        # community volume = sum of strengths in S
        vol = sum(strength.get(u, 0.0) for u in S)

        Q += (w_in / m) - (vol / two_m) ** 2

    return Q


def compute_conductance(G, clusters, average='mean', return_per_cluster=False):
    """
    Weighted conductance for each cluster in a hard partition.
    clusters: dict {cluster_id -> iterable_of_nodes}
    For a set S:
      φ(S) = cut(S, ~S) / min(vol(S), vol(~S))
      - cut: sum of weights of edges crossing S and its complement
      - vol: sum of strengths (weighted degrees) of nodes in the set
    Returns:
      - if return_per_cluster=False: a single scalar (mean/median over communities)
      - if return_per_cluster=True: (scalar, {cid: φ})
    """
    m = G.size(weight='weight')
    if m == 0:
        return (0.0, {}) if return_per_cluster else 0.0

    strength = dict(G.degree(weight='weight'))
    two_m = 2.0 * m

    cond = {}

    for cid, nodes in clusters.items():
        S = set(nodes)
        if not S:
            cond[cid] = 0.0
            continue

        vol_S = sum(strength.get(u, 0.0) for u in S)
        vol_comp = two_m - vol_S

        # cut weight: sum of weights of edges with one end in S and the other not in S
        cut = 0.0
        for u in S:
            for v, data in G[u].items():
                if v not in S:
                    cut += data.get('weight', 1.0)

        denom = min(vol_S, vol_comp)
        phi = 0.0 if denom <= 0.0 else (cut / denom)
        cond[cid] = phi

    vals = list(cond.values())
    if not vals:
        overall = 0.0
    elif average == 'median':
        vals_sorted = sorted(vals)
        mid = len(vals_sorted) // 2
        overall = (vals_sorted[mid] if len(vals_sorted) % 2 == 1
                   else 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid]))
    else:
        overall = sum(vals) / len(vals)

    if return_per_cluster:
        return overall, cond
    return overall




def compute_distance(G, u, v):
    try:
        return nx.shortest_path_length(G, u, v, weight='weight')
    except nx.NetworkXNoPath:
        return float('inf')

def compute_avg_distance_to_cluster(G, node, cluster_nodes):
    if not cluster_nodes or node not in G:
        return float('inf')
    
    distances = []
    for cluster_node in cluster_nodes:
        if cluster_node != node:
            dist = compute_distance(G, node, cluster_node)
            if dist != float('inf'):
                distances.append(dist)
    
    return np.mean(distances) if distances else float('inf')


######## DBI
def compute_DBI(G, clusters):
    
    if len(clusters) <= 1:
        return float('inf')
    
    def compute_diameter(nodes):
        node_list = list(nodes)
        if len(node_list) <= 1:
            return 0.0
        
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                u, v = node_list[i], node_list[j]
                distance = compute_distance(G, u, v)
                
                if distance != float('inf'):
                    total_distance += distance
                    pair_count += 1
        
        return total_distance / pair_count if pair_count > 0 else 0.0
    
    def compute_inter_cluster_distance(nodes_i, nodes_j):
        total_distance = 0.0
        pair_count = 0
        
        for u in nodes_i:
            for v in nodes_j:
                distance = compute_distance(G, u, v)
                
                if distance != float('inf'):
                    total_distance += distance
                    pair_count += 1
        
        return total_distance / pair_count if pair_count > 0 else float('inf')
    
    cluster_ids = list(clusters.keys())
    diameters = {cid: compute_diameter(clusters[cid]) for cid in cluster_ids}
    
    dbi_sum = 0.0
    k = len(cluster_ids)
    
    for i, cluster_i in enumerate(cluster_ids):
        max_ratio = 0.0
        
        for j, cluster_j in enumerate(cluster_ids):
            if i != j:
                numerator = diameters[cluster_i] + diameters[cluster_j]
                denominator = compute_inter_cluster_distance(clusters[cluster_i], clusters[cluster_j])
                
                if denominator > 0:
                    ratio = numerator / denominator
                    max_ratio = max(max_ratio, ratio)
        
        dbi_sum += max_ratio
    
    return dbi_sum / k


######## SI
def compute_SI(G, clusters):
    
    if len(clusters) <= 1:
        return -1.0
    
    cluster_silhouettes = []
    
    for cluster_id, cluster_nodes in clusters.items():
        cluster_nodes = list(cluster_nodes)
        if len(cluster_nodes) <= 1:
            continue
        
        node_silhouettes = []
        
        for node in cluster_nodes:
            a_vi = compute_avg_distance_to_cluster(G, node, cluster_nodes)
            
            b_vi = float('inf')
            
            for other_cluster_id, other_cluster_nodes in clusters.items():
                if other_cluster_id != cluster_id:
                    other_cluster_nodes = list(other_cluster_nodes)
                    if len(other_cluster_nodes) > 0:
                        avg_dist = compute_avg_distance_to_cluster(G, node, other_cluster_nodes)
                        b_vi = min(b_vi, avg_dist)
            
            if a_vi == float('inf') or b_vi == float('inf'):
                s_vi = 0.0
            elif a_vi == 0.0 and b_vi == 0.0:
                s_vi = 0.0
            else:
                s_vi = (b_vi - a_vi) / max(a_vi, b_vi)
            
            node_silhouettes.append(s_vi)
        
        if node_silhouettes:
            cluster_avg_silhouette = np.mean(node_silhouettes)
            cluster_silhouettes.append(cluster_avg_silhouette)
    
    if cluster_silhouettes:
        return np.mean(cluster_silhouettes)
    else:
        return -1.0
    
def compute_Qs(G, clusters, similarity_func, gamma):
    if len(clusters) == 0:
        return 0.0
    
    TS = 0.0
    all_nodes = list(G.nodes())
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            u, v = all_nodes[i], all_nodes[j]
            sim = similarity_func(G, u, v, gamma)
            if not np.isnan(sim) and not np.isinf(sim):
                TS += sim
    
    if TS == 0.0:
        return 0.0
    
    Qs = 0.0
    
    for cluster_id, cluster_nodes in clusters.items():
        cluster_nodes = list(set(cluster_nodes))
        
        # ISi
        ISi = 0.0
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                u, v = cluster_nodes[i], cluster_nodes[j]
                sim = similarity_func(G, u, v, gamma)
                if not np.isnan(sim) and not np.isinf(sim):
                    ISi += sim
        
        # DSi
        DSi = 0.0
        cluster_nodes_set = set(cluster_nodes)
        for u in cluster_nodes:
            for v in all_nodes:
                if u != v:
                    sim = similarity_func(G, u, v, gamma)
                    if not np.isnan(sim) and not np.isinf(sim):
                        if v in cluster_nodes_set and u < v:
                            DSi += sim
                        elif v not in cluster_nodes_set:
                            DSi += sim
        
        # Qs += ISi/TS - (DSi/TS)^2
        if TS > 0:
            Qs += (ISi / TS) - (DSi / TS) ** 2
    
    return Qs