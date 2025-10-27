import argparse
import time
import networkx as nx
import psutil
import os

import clustering
import similarity

from metrics import compute_ARI, compute_NMI
from metrics import compute_modularity, compute_conductance, compute_DBI, compute_SI, compute_Qs
from utils import load_ground_truth, save_result_to_csv, save_result_to_csv_no_gt
from change_graph import perturb_edge_weights

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

parser.add_argument("--similarity", choices=["SCAN", "WSCAN", "cosine", "WSCAN++", "WSCAN++_max", "WSCAN++_avg", "Jaccard"],
                    default="WSCAN++",
                    help="choose similarity function")
parser.add_argument("--network", default="karate",
                    help="path to weighted edge list (u v w)")
parser.add_argument("--gt", default="True",
                    help="does ground truth exist? (True or False)")

parser.add_argument("--dataclass", default="real",
                    help="is dataset real? real or synthetic")
parser.add_argument("--exp_mode", default="time",
                    help="exp for what? effectiveness or time")

parser.add_argument("--delta_p", type=float, default=0,
                    help="how much percent of max_w will be delta")
parser.add_argument("--edge_p", type=float, default=0,
                    help="how much percent of edges will be changed")
parser.add_argument("--weight_method", default="max",
                    help="which weight will be multiplied to delta (max or avg)")

parser.add_argument("--use_parallel", default="False",
                    help="using parallel or not (True or False)")
parser.add_argument("--process_num", type=int, default=1,
                    help="number of processes for parallel")

args = parser.parse_args()

process = psutil.Process(os.getpid())
memory_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB

path = "../dataset/" + args.dataclass + "/"
if args.dataclass == "synthetic":
      path += args.exp_mode + "/scale/"
# --------------------------------------------------------------------
# load network  (expects 'u v weight' per line)
# --------------------------------------------------------------------
dataset = path + args.network + "/network.dat"
G = nx.read_weighted_edgelist(dataset, nodetype=int)

# print(len(G.nodes))
# print(len(G.edges))

print(f"Loaded graph: {args.network}  "
      f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

# --------------------------------------------------------------------
# Change graph edge weights
# --------------------------------------------------------------------
perturb_edge_weights(G, args.delta_p, args.edge_p, args.weight_method)

# --------------------------------------------------------------------
# load answer
# --------------------------------------------------------------------
gt_filename = "community" if args.dataclass == "synthetic" else "labels"
gt_path = path + args.network + "/" + gt_filename + ".dat"
if args.gt == "True":
      ground_truth = load_ground_truth(gt_path)

# --------------------------------------------------------------------
# run selected algorithm
# --------------------------------------------------------------------
sim = {"SCAN" : similarity.scan_similarity, "WSCAN" : similarity.wscan_similarity,
       "cosine" : similarity.cosine_similarity, "WSCAN++" : similarity.wscan_p_similarity,
       "WSCAN++_max" : similarity.wscan_p_similarity_max, "WSCAN++_avg" : similarity.wscan_p_similarity_avg,
       "Jaccard" : similarity.weighted_jaccard_similarity}

similarity_func = sim[args.similarity]

parallel = True if args.use_parallel == "True" else False

if "WSCAN++" in args.similarity:
      start = time.time()
      clusters, hubs, outliers, similarity_calculating_time = clustering.run(G, similarity_func, eps=args.eps, mu=args.mu, gamma=args.gamma, parallel=parallel, workers=args.process_num)
      runtime = time.time() - start
else:
      start = time.time()
      clusters, hubs, outliers, similarity_calculating_time = clustering.run_include_me(G, similarity_func, eps=args.eps, mu=args.mu, gamma=args.gamma, parallel=parallel, workers=args.process_num)
      runtime = time.time() - start

memory_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
memory_usage = memory_after - memory_before  # Calculate memory used

# --------------------------------------------------------------------
# basic report
# --------------------------------------------------------------------
# print(hubs)
# print(outliers)
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

if args.gt == "True":
      ari_score = compute_ARI(clusters, ground_truth)
      print(f"Adjusted Rand Index: {ari_score:.4f}")
      nmi_score = compute_NMI(clusters, ground_truth)
      print(f"NMI: {nmi_score:.4f}")
else:
      ari_score = None
      nmi_score = None

# --- counts ---
num_clusters = len(clusters)
num_hubs = len(hubs)
num_outliers = len(outliers)

# plot_clusters(G, clusters)

args.output_path = "./exp/" + args.exp_mode + "/test.csv"

save_result_to_csv(args, runtime, memory_usage, ari_score, nmi_score, 
            num_clusters, num_hubs, num_outliers, similarity_calculating_time)