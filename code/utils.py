import matplotlib.pyplot as plt
import networkx as nx

import csv
import os

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


def save_result_to_csv(args, runtime, memory_usage, ari_score, nmi_score):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    write_header = not os.path.exists(args.output_path)
    dataset_name = os.path.basename(os.path.dirname(args.network))

    with open(args.output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "dataset", "similarity", "eps", "mu", "gamma",
                "runtime(sec)", "memory(MB)", "ARI", "NMI"
            ])

        writer.writerow([
            dataset_name,
            args.similarity,
            args.eps,
            args.mu,
            args.gamma,
            f"{runtime:.4f}",
            f"{memory_usage:.2f}",
            ari_score if ari_score is not None else "Non",
            nmi_score if nmi_score is not None else "Non"
        ])
    print(f"Saved results to {args.output_path}")