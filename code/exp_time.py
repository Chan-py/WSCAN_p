# Usage example:
#   python main.py --network collegemsg \
#                  --eps 0.25 --mu 2 --similarity Gen

import argparse
import time
import networkx as nx
import psutil
import os

import clustering
import similarity

from metrics import compute_ARI, compute_NMI
from utils import load_ground_truth, plot_clusters, save_result_to_csv

# --------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Weighted-SCAN runner")

parser.add_argument("--eps", type=float, default=0.5,
                    help="ε similarity threshold")
parser.add_argument("--mu",  type=int,   default=2,
                    help="minimum number of ε-neighbors (core)")
parser.add_argument("--gamma",  type=float,   default=1,
                    help="degree considering undirected edges")

parser.add_argument("--similarity", choices=["scan", "wscan", "cosine", "Gen", "Jaccard"],
                    default="Gen",
                    help="choose similarity function")
parser.add_argument("--network", default="karate",
                    help="path to weighted edge list (u v w)")
parser.add_argument("--gt", default=False,
                    help="path to ground truth")

args = parser.parse_args()

process = psutil.Process(os.getpid())
memory_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB

# --------------------------------------------------------------------
# load network  (expects 'u v weight' per line)
# --------------------------------------------------------------------
dataset = "../dataset/synthetic/scalability/" + args.network + ".dat"
G = nx.read_weighted_edgelist(dataset, nodetype=int)

# print(len(G.nodes))
# print(len(G.edges))

print(f"Loaded graph: {args.network}  "
      f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

# --------------------------------------------------------------------
# load answer
# --------------------------------------------------------------------
gt_path = "../dataset/synthetic/scalability/" + args.network + ".gt"
if args.gt:
      ground_truth = load_ground_truth(gt_path)

# --------------------------------------------------------------------
# run selected algorithm
# --------------------------------------------------------------------
sim = {"scan" : similarity.scan_similarity, "wscan" : similarity.wscan_similarity,
       "cosine" : similarity.cosine_similarity, "Gen" : similarity.Gen_wscan_similarity, 
       "Jaccard" : similarity.weighted_jaccard_similarity}

similarity_func = sim[args.similarity]

start = time.time()
clusters, hubs, outliers = clustering.run(G, similarity_func, eps=args.eps, mu=args.mu, gamma=args.gamma)
runtime = time.time() - start

memory_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
memory_usage = memory_after - memory_before  # Calculate memory used

# --------------------------------------------------------------------
# basic report
# --------------------------------------------------------------------
print("\n=== RESULT SUMMARY ===")
print(f"similarity        : {args.similarity}")
print(f"ε, μ              : {args.eps}, {args.mu}")
print(f"runtime (seconds) : {runtime:.3f}")
print(f"#clusters         : {len(clusters)}")
print(f"#hubs             : {len(hubs)}")
print(f"#outliers         : {len(outliers)}")

for cid, nodes in list(clusters.items()):
    print(f"  cluster {cid:<2} size={len(nodes)} nodes={list(nodes)}")

# --------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------

if args.gt:
      ari_score = compute_ARI(clusters, ground_truth)
      print(f"Adjusted Rand Index: {ari_score:.4f}")
      nmi_score = compute_NMI(clusters, ground_truth)
      print(f"NMI: {nmi_score:.4f}")
else:
      ari_score = None
      nmi_core = None

# modularity = compute_modularity(G, clusters)
# DBI = compute_DBI(G, clusters)
# SI = compute_SI(G, clusters)
# Qs = compute_Qs(G, clusters, similarity_func, args.gamma)

plot_clusters(G, clusters)

args.output_path = "./output/time.csv"
save_result_to_csv(args, runtime, memory_usage, ari_score, nmi_score)