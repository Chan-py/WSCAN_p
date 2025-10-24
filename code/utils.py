# Spring layout + matplotlib 예시
import matplotlib.pyplot as plt
import networkx as nx

def plot_clusters(G, clusters):
    pos = nx.spring_layout(G, weight='weight')
    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(8,8))
    for cid, nodes in clusters.items():
        nx.draw_networkx_nodes(G, pos,
                               nodelist=list(nodes),
                               node_size=50,
                               node_color=[cmap(cid % 20)])
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.axis('off')
    plt.show()

def load_ground_truth(labels_path):
    """label.dat 포맷: node true_label (공백 구분)"""
    gt = {}
    with open(labels_path, 'r') as f:
        for line in f:
            node, label = line.strip().split()
            gt[int(node)] = label  # label이 int면 int(label)
    return gt

import csv
import os

def save_result_to_csv(args, runtime, memory_usage, ari_score, nmi_score,
                       num_clusters, num_hubs, num_outliers, similarity_calculating_time):

    # output 폴더가 없으면 생성
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    write_header = not os.path.exists(args.output_path)

    # dataset_name 설정 (여기서는 인자로 받은 네트워크 이름을 그대로 사용)
    dataset_name = args.network

    with open(args.output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "dataset", "similarity", "eps", "mu", "gamma", "similarity_computing_time",
                "runtime(sec)", "memory(MB)", "ARI", "NMI",
                "num_clusters", "num_hubs", "num_outliers",
                "delta_p", "edge_p", "perterb_weight_method",
                "is_parallel", "num_processes"
            ])

        writer.writerow([
            dataset_name,
            args.similarity,
            args.eps,
            args.mu,
            args.gamma,
            f"{similarity_calculating_time:.4f}",
            f"{runtime:.4f}",
            f"{memory_usage:.2f}",
            ari_score if ari_score is not None else "Non",
            nmi_score if nmi_score is not None else "Non",
            num_clusters,
            num_hubs,
            num_outliers,
            args.delta_p,
            args.edge_p,
            args.weight_method,
            args.use_parallel,
            args.process_num
        ])

    print(f"Saved results to {args.output_path}")


import numpy as np

def eps_grid_by_quantiles(G, similarity_func, num_quantiles=17, quantile_range=(0.1, 0.9)):
    # 모든 엣지에 대해 similarity 값 계산
    similarity_values = []
    
    for u, v in G.edges():
        sim_value = similarity_func(G, u, v, 1)
        if not np.isnan(sim_value) and not np.isinf(sim_value):
            similarity_values.append(sim_value)
    
    if not similarity_values:
        raise ValueError("유효한 similarity 값을 계산할 수 없습니다.")
    
    similarity_values = np.array(similarity_values)
    
    # quantile 기반으로 eps 값들 생성
    min_quantile, max_quantile = quantile_range
    quantiles = np.linspace(min_quantile, max_quantile, num_quantiles)
    eps_candidates = np.quantile(similarity_values, quantiles)
    
    # 중복 제거 및 정렬
    eps_candidates = np.unique(eps_candidates)
    
    return eps_candidates