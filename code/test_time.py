import networkx as nx
from itertools import combinations
import time

import similarity
import clustering

from utils import load_ground_truth, eps_grid_by_quantiles
from metrics import compute_ARI, compute_NMI, compute_modularity, compute_DBI, compute_SI, compute_Qs
from parameter import estimate_eps_kneedle

times = ["20K", "40K", "80K"]
# times = ["20K", "40K", "80K", "160K", "320K"]
for t in times:
    edges_file = "../dataset/synthetic/scalability/LFR_" + t + ".dat"
    labels_file = "../dataset/synthetic/scalability/LFR_" + t + ".gt"
    G = nx.read_weighted_edgelist(edges_file, nodetype=int)
    ground_truth = load_ground_truth(labels_file)

    sim = {"scan" : similarity.scan_similarity, "wscan" : similarity.wscan_similarity,
       "cosine" : similarity.cosine_similarity, "wscan_p" : similarity.wscan_p_similarity, 
       "Jaccard" : similarity.weighted_jaccard_similarity}
    
    for s in sim:
        similarity_func = sim[s]

        mu = 4

        start_1 = time.time()
        e= estimate_eps_kneedle(G, similarity_func=similarity_func, mu=mu, gamma=1, plot = False)
        print("time: ", t, ", similarity func: ", s)
        print("parameter detection")
        print(time.time() - start_1)

        start_2 = time.time()
        clusters, hubs, outliers = clustering.run(G, similarity_func, eps=e, mu=mu, gamma=1)
        print("algorithm running time")
        print(time.time() - start_2)