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

    미소속 노드(허브/아웃라이어)는 예측 라벨 -1로 처리합니다.
    """
    # 예측 라벨 매핑
    y_pred_map = {}
    for cid, members in clusters.items():
        for u in members:
            y_pred_map[u] = cid

    # 비교할 노드 목록(ground truth에 정의된 노드 기준)
    nodes = sorted(ground_truth.keys())
    y_true = [ground_truth[u] for u in nodes]
    y_pred = [y_pred_map.get(u, -1) for u in nodes]

    return normalized_mutual_info_score(
        y_true, y_pred, average_method=average_method
    )





######### Modularity
def compute_modularity(G, clusters):
    """
    Compute modularity for an undirected weighted graph G and given clusters.
    clusters: dict mapping cluster_id -> iterable of nodes
    Returns Q = sum over clusters of [E_C/m - (sum_deg_C/(2m))^2]
    """
    # Total edge weight (each undirected edge counted once)
    m = sum(data['weight'] for u, v, data in G.edges(data=True))
    Q = 0.0

    for cid, nodes in clusters.items():
        node_set = set(nodes)
        # Sum of weights of edges inside the cluster
        E_C = 0.0
        # Sum of degrees (strengths) of nodes in the cluster
        sum_deg_C = 0.0

        # Compute E_C
        for u in node_set:
            for v, data in G[u].items():
                if v in node_set:
                    E_C += data['weight']
        E_C /= 2.0  # each internal edge counted twice in the loop

        # Compute sum_deg_C
        for u in node_set:
            sum_deg_C += sum(data['weight'] for _, _, data in G.edges(u, data=True))

        # Community-level modularity contribution
        Q += (E_C / m) - (sum_deg_C / (2 * m))**2

    return Q


########### NMI
def compute_NMI_(clusters, ground_truth, average_method='arithmetic'):
    """
    Compute Normalized Mutual Information (NMI).
      - clusters: dict {cluster_id -> iterable_of_nodes}
      - ground_truth: dict {node -> true_label}
      - average_method: {'min', 'geometric', 'arithmetic', 'max'} (sklearn 옵션)

    미소속 노드(허브/아웃라이어)는 예측 라벨 -1로 처리합니다.
    """
    # 예측 라벨 매핑
    y_pred_map = {}
    for cid, members in clusters.items():
        for u in members:
            y_pred_map[u] = cid

    # 비교할 노드 목록(ground truth에 정의된 노드 기준)
    nodes = sorted(ground_truth.keys())
    y_true = [ground_truth[u] for u in nodes]
    y_pred = [y_pred_map.get(u, -1) for u in nodes]

    return normalized_mutual_info_score(
        y_true, y_pred, average_method=average_method
    )


def compute_distance(G, u, v):
    """두 노드 간의 거리 계산 (최단경로)"""
    try:
        return nx.shortest_path_length(G, u, v, weight='weight')
    except nx.NetworkXNoPath:
        return float('inf')

def compute_avg_distance_to_cluster(G, node, cluster_nodes):
    """한 노드에서 클러스터 내 모든 노드들까지의 평균 거리"""
    if not cluster_nodes or node not in G:
        return float('inf')
    
    distances = []
    for cluster_node in cluster_nodes:
        if cluster_node != node:  # 자기 자신 제외
            dist = compute_distance(G, node, cluster_node)
            if dist != float('inf'):
                distances.append(dist)
    
    return np.mean(distances) if distances else float('inf')


######## DBI
def compute_DBI(G, clusters):
    
    if len(clusters) <= 1:
        return float('inf')
    
    # 각 클러스터의 diameter 계산
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
    
    # 두 클러스터 간의 거리 계산
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
    
    # 각 클러스터의 diameter 계산
    cluster_ids = list(clusters.keys())
    diameters = {cid: compute_diameter(clusters[cid]) for cid in cluster_ids}
    
    # DBI 계산
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
        return -1.0  # 클러스터가 1개 이하면 silhouette 계산 불가
    
    cluster_silhouettes = []
    
    for cluster_id, cluster_nodes in clusters.items():
        cluster_nodes = list(cluster_nodes)
        if len(cluster_nodes) <= 1:
            continue  # 노드가 1개인 클러스터는 건너뜀
        
        node_silhouettes = []
        
        for node in cluster_nodes:
            # a(vi): 같은 클러스터 내 다른 노드들과의 평균 거리
            a_vi = compute_avg_distance_to_cluster(G, node, cluster_nodes)
            
            # b(vi): 가장 가까운 다른 클러스터와의 평균 거리
            b_vi = float('inf')
            
            for other_cluster_id, other_cluster_nodes in clusters.items():
                if other_cluster_id != cluster_id:
                    other_cluster_nodes = list(other_cluster_nodes)
                    if len(other_cluster_nodes) > 0:
                        avg_dist = compute_avg_distance_to_cluster(G, node, other_cluster_nodes)
                        b_vi = min(b_vi, avg_dist)
            
            # s(vi) 계산
            if a_vi == float('inf') or b_vi == float('inf'):
                s_vi = 0.0
            elif a_vi == 0.0 and b_vi == 0.0:
                s_vi = 0.0
            else:
                s_vi = (b_vi - a_vi) / max(a_vi, b_vi)
            
            node_silhouettes.append(s_vi)
        
        # 클러스터의 평균 silhouette
        if node_silhouettes:
            cluster_avg_silhouette = np.mean(node_silhouettes)
            cluster_silhouettes.append(cluster_avg_silhouette)
    
    # 전체 Silhouette Index (모든 클러스터의 평균)
    if cluster_silhouettes:
        return np.mean(cluster_silhouettes)
    else:
        return -1.0
    
def compute_Qs(G, clusters, similarity_func, gamma):
    if len(clusters) == 0:
        return 0.0
    
    # TS: 전체 네트워크의 total similarity (모든 노드 쌍에 대해)
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
    
    # 각 클러스터별로 ISi와 DSi 계산
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
                if u != v:  # 자기 자신 제외
                    sim = similarity_func(G, u, v, gamma)
                    if not np.isnan(sim) and not np.isinf(sim):
                        # 같은 클러스터 내 노드들 간의 similarity는 절반만 계산 (중복 방지)
                        if v in cluster_nodes_set and u < v:
                            DSi += sim  # u < v 조건으로 한 번만 계산
                        elif v not in cluster_nodes_set:
                            DSi += sim  # 다른 클러스터나 hub/outlier와는 모두 계산
        
        # Qs에 각 클러스터의 기여도 추가
        # Qs += ISi/TS - (DSi/TS)^2
        if TS > 0:
            Qs += (ISi / TS) - (DSi / TS) ** 2
    
    return Qs