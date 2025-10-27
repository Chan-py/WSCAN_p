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
    """label.dat format: node true_label"""
    gt = {}
    with open(labels_path, 'r') as f:
        for line in f:
            node, label = line.strip().split()
            gt[int(node)] = label
    return gt

import csv
import os

def save_result_to_csv(args, runtime, memory_usage, ari_score, nmi_score,
                       num_clusters, num_hubs, num_outliers, similarity_calculating_time):

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    write_header = not os.path.exists(args.output_path)

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