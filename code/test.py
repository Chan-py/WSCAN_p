import networkx as nx
from itertools import combinations

import similarity
import clustering

from utils import load_ground_truth, eps_grid_by_quantiles
from metrics import compute_ARI, compute_NMI, compute_modularity, compute_DBI, compute_SI, compute_Qs
from parameter import estimate_eps_kneedle

# G = nx.read_weighted_edgelist("../dataset/real/ca-cit-HepTh/network.dat", nodetype=int)     # 22908       너무 오래 걸림
# G = nx.read_weighted_edgelist("../dataset/real/sociopatterns-infectious/network.dat", nodetype=int)   # 410 Pass
# G = nx.read_weighted_edgelist("../dataset/real/moreno_names/network.dat", nodetype=int)     # 1773         다 너무 낮음
# G = nx.read_weighted_edgelist("../dataset/real/dnc-corecipient/network.dat", nodetype=int)     # 906     다 너무 낮음 낫배드

# G = nx.read_weighted_edgelist("../dataset/real/les_miserable/network.dat", nodetype=int)      # 58
# G = nx.read_weighted_edgelist("../dataset/real/collegemsg/network.dat", nodetype=int)         # 1203
# G = nx.read_weighted_edgelist("../dataset/real/collegemsg_edges.dat", nodetype=int) 

network = "sociopattern_workplace_v2"
dataset = "../dataset/real/" + network + "/network.dat"
G = nx.read_weighted_edgelist(dataset, nodetype=int)
gt_path = "../dataset/real/" + network + "/labels.dat"
ground_truth = load_ground_truth(gt_path)

print(f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

sim = {"scan" : similarity.scan_similarity, "wscan" : similarity.wscan_similarity,
       "cosine" : similarity.cosine_similarity, "wscan_p" : similarity.wscan_p_similarity, 
       "Jaccard" : similarity.weighted_jaccard_similarity}

import numpy as np

for s in sim:
       similarity_func = sim[s]
       # if s != "Gen":
       #        continue

       best_ari = -1
       best_ari_params = (0, 0, 0)

       best_nmi = -1
       best_nmi_params = (0, 0, 0)

       # max_mod = -1.0
       # max_parm_mod = (0, 0, 0)

       # # DBI: 낮을수록 좋음
       # best_dbi = float('inf')
       # best_parm_dbi = (0, 0, 0)

       # # Silhouette: 높을수록 좋음
       # best_si = -1.0
       # best_parm_si = (0, 0, 0)

       # # Similarity-based Modularity Qs: 높을수록 좋음
       # best_qs = -1.0
       # best_parm_qs = (0, 0, 0)


       eps_candidates = np.arange(0.05, 1, 0.05)
       if s == "wscan":
              eps_candidates = eps_grid_by_quantiles(G, similarity_func)
       
       if s == "Gen":
              gamma_range = np.arange(0.5, 1.05, 0.1)
       else:
              gamma_range = range(1, 2)
       for mu in range(2, 8):
              for gamma in gamma_range:
                     e= estimate_eps_kneedle(G, similarity_func=similarity_func, mu=mu, gamma=gamma, plot = False)
              
                     clusters, hubs, outliers = clustering.run(G, similarity_func, eps=e, mu=mu, gamma=gamma)

                     ari = compute_ARI(clusters, ground_truth)
                     # if ari != 0:
                            # print(f"ε={e:.2f}, μ={mu:2d}, gamma={gamma:.2f} -> ARI={ari:.4f}")
                     if ari > best_ari:
                            best_ari = ari
                            best_ari_params = (e, mu, gamma)

                     nmi = compute_NMI(clusters, ground_truth)
                     # if nmi != 0:
                            # print(f"ε={e:.2f}, μ={mu:2d}, gamma={gamma:.2f} -> NMI={nmi:.4f}")
                     if nmi > best_nmi:
                            best_nmi = nmi
                            best_nmi_params = (e, mu, gamma)

                     # # Modularity
                     # modularity = compute_modularity(G, clusters)
                     # if modularity > max_mod:
                     #        max_mod = modularity
                     #        max_parm_mod = (e, mu, gamma)

                     # # DBI (낮을수록 좋음)
                     # dbi = compute_DBI(G, clusters)
                     # if dbi < best_dbi:
                     #        best_dbi = dbi
                     #        best_parm_dbi = (e, mu, gamma)

                     # # Silhouette (높을수록 좋음)
                     # si = compute_SI(G, clusters)
                     # if si > best_si:
                     #        best_si = si
                     #        best_parm_si = (e, mu, gamma)

                     # # Similarity-based Modularity Qs (높을수록 좋음)
                     # qs = compute_Qs(G, clusters, similarity_func, 1)
                     # if qs > best_qs:
                     #        best_qs = qs
                     #        best_parm_qs = (e, mu, gamma)

                     # print(f"[{s}] ε={e:.2f} -> Mod={modularity:.4f}, DBI={dbi:.4f}, SI={si:.4f}, Qs={qs:.4f}")

       print("======== Max/Min over", s)
       print(f"(ARI) ε={best_ari_params[0]:.2f}, μ={best_ari_params[1]:2d}, gamma={best_ari_params[2]:.1f} -> {best_ari:.4f}")
       print(f"(NMI) ε={best_nmi_params[0]:.2f}, μ={best_nmi_params[1]:2d}, gamma={best_nmi_params[2]:.1f} -> {best_nmi:.4f}")

       # print(f"(Mod) ε={max_parm_mod[0]:.2f}, μ={max_parm_mod[1]:2d}, gamma={max_parm_mod[2]:.1f} -> {max_mod:.4f}")
       # print(f"(DBI) ε={best_parm_dbi[0]:.2f}, μ={best_parm_dbi[1]:2d}, gamma={best_parm_dbi[2]:.1f} -> {best_dbi:.4f}  (lower is better)")
       # print(f"(SI ) ε={best_parm_si[0]:.2f}, μ={best_parm_si[1]:2d}, gamma={best_parm_si[2]:.1f} -> {best_si:.4f}")
       # print(f"(Qs ) ε={best_parm_qs[0]:.2f}, μ={best_parm_qs[1]:2d}, gamma={best_parm_qs[2]:.1f} -> {best_qs:.4f}")
       print()
              