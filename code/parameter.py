
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import networkx as nx
from typing import Callable, Optional, Tuple


def estimate_eps_kneedle(G: nx.Graph, 
                        similarity_func: Callable = None,
                        mu: int = 6, 
                        gamma: float = 0.5,
                        plot: bool = True,
                        curve: str = 'convex',
                        direction: str = 'decreasing',
                        sensitivity: float = 1.0) -> float:
    
    mu_similarities = []
    
    # Calculate Mu-th biggest similarity for each node
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        
        if len(neighbors) == 0:
            mu_similarities.append(0.0)
            continue
            
        # Compute similarity with every neighbors
        similarities = []
        for neighbor in neighbors:
            sim = similarity_func(G, node, neighbor, gamma)
            similarities.append(sim)
        
        # Sort similarities
        similarities.sort(reverse=True)
        
        # Mu-th similarity
        if len(similarities) >= mu:
            mu_sim = similarities[mu-1]
        else:
            mu_sim = similarities[-1] if similarities else 0.0
            
        mu_similarities.append(mu_sim)
    
    # Sort Mu-th similarities
    sorted_similarities = sorted(mu_similarities, reverse=True)
    
    # kneedle point
    if len(sorted_similarities) < 3:
        # If not enough datas
        eps_estimate = np.median(sorted_similarities)
        print(f"Warning: Not enough data -> Using median. eps = {eps_estimate:.4f}")
        return eps_estimate
    
    x = np.arange(len(sorted_similarities))
    y = np.array(sorted_similarities)
    
    try:
        # Various ways of kneedle point
        knee_candidates = []
        
        # Tactic 1
        try:
            kneedle1 = KneeLocator(x, y, 
                                 curve='convex', 
                                 direction='decreasing',
                                 S=sensitivity*0.3)
            if kneedle1.knee is not None:
                knee_candidates.append(('kneedle_sensitive', kneedle1.knee))
        except:
            pass
            
        # Tactic 2
        try:
            kneedle2 = KneeLocator(x, y, 
                                 curve='concave', 
                                 direction='decreasing',
                                 S=sensitivity*0.2)
            if kneedle2.knee is not None:
                knee_candidates.append(('kneedle_concave', kneedle2.knee))
        except:
            pass
        
        # Tactic 3
        if len(y) > 3:
            gradients = np.diff(y)

            start_idx = max(1, int(len(gradients) * 0.1))
            end_idx = min(len(gradients), int(len(gradients) * 0.8))
            
            if end_idx > start_idx:
                search_gradients = gradients[start_idx:end_idx]
                steepest_relative_idx = np.argmin(search_gradients)
                steepest_idx = start_idx + steepest_relative_idx
                knee_candidates.append(('gradient_based', steepest_idx))
        
        # Tactic 4
        q40_idx = int(len(sorted_similarities) * 0.4)
        q80_idx = int(len(sorted_similarities) * 0.8)
        
        if q80_idx > q40_idx + 2:
            mid_gradients = np.diff(y[q40_idx:q80_idx])
            if len(mid_gradients) > 0:
                mid_steepest_idx = np.argmin(mid_gradients)
                knee_candidates.append(('quantile_based', q40_idx + mid_steepest_idx))
        
        # Select the best
        if knee_candidates:
            # Filtering
            min_acceptable_idx = int(len(sorted_similarities) * 0.2)
            valid_candidates = [(name, idx) for name, idx in knee_candidates 
                              if idx >= min_acceptable_idx]
            
            if valid_candidates:
                knee_method, knee_point = min(valid_candidates, key=lambda x: x[1])
                eps_estimate = sorted_similarities[knee_point]
                # print(f"Knee point found using {knee_method} at index {knee_point}, eps = {eps_estimate:.4f}")
            else:
                knee_point = int(len(sorted_similarities) * 0.4)
                eps_estimate = sorted_similarities[knee_point]
                print(f"Using 40% quantile as eps = {eps_estimate:.4f}")
        else:
            knee_point = int(len(sorted_similarities) * 0.5)
            eps_estimate = sorted_similarities[knee_point]
            print(f"Fallback: Using median position, eps = {eps_estimate:.4f}")
            
    except Exception as e:
        print(f"Error in computing Kneedle: {e}")
        knee_point = int(len(sorted_similarities) * 0.5)
        eps_estimate = sorted_similarities[knee_point]
        print(f"Error fallback: eps = {eps_estimate:.4f}")
    
    # Plot
    if plot:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'b-', linewidth=2, label='Sorted μ-th similarities (descending)')
        if knee_point is not None:
            plt.axvline(x=knee_point, color='r', linestyle='--', 
                       label=f'Kneedle point (eps={eps_estimate:.4f})')
            plt.plot(knee_point, sorted_similarities[knee_point], 
                    'ro', markersize=8, label='Eps estimate')
        plt.xlabel('Node index (sorted by similarity desc)')
        plt.ylabel('μ-th largest similarity')
        plt.title(f'μ-th Similarities Distribution (μ={mu})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if len(y) > 1:
            ax2 = plt.twinx()
            gradients = np.diff(y)
            ax2.plot(x[:-1], gradients, 'g--', alpha=0.5, label='Gradient')
            ax2.set_ylabel('Gradient', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            q20_idx = int(len(sorted_similarities) * 0.2)
            q40_idx = int(len(sorted_similarities) * 0.4)
            q80_idx = int(len(sorted_similarities) * 0.8)
            
            plt.axvspan(q20_idx, q40_idx, alpha=0.1, color='yellow', label='Search region 1')
            plt.axvspan(q40_idx, q80_idx, alpha=0.1, color='orange', label='Search region 2')
        
        plt.subplot(1, 2, 2)
        plt.hist(sorted_similarities, bins=min(30, len(sorted_similarities)//2), 
                alpha=0.7, edgecolor='black')
        plt.axvline(x=eps_estimate, color='r', linestyle='--', linewidth=2,
                   label=f'Eps estimate = {eps_estimate:.4f}')
        plt.xlabel('μ-th similarity value')
        plt.ylabel('Frequency')
        plt.title('Distribution of μ-th Similarities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Statistics
        print(f"\n=== Eps Estimation Results ===")
        print(f"Graph nodes: {G.number_of_nodes()}")
        print(f"Graph edges: {G.number_of_edges()}")
        print(f"μ parameter: {mu}")
        print(f"Estimated eps: {eps_estimate:.6f}")
        print(f"Min μ-th similarity: {min(sorted_similarities):.6f}")
        print(f"Max μ-th similarity: {max(sorted_similarities):.6f}")
        print(f"Mean μ-th similarity: {np.mean(sorted_similarities):.6f}")
        print(f"Median μ-th similarity: {np.median(sorted_similarities):.6f}")
    
    return eps_estimate


# Example

if __name__ == "__main__":

    import networkx as nx
    from utils import load_ground_truth
    import similarity

    network = "karate"
    dataset = "../dataset/real/" + network + "/network.dat"
    G = nx.read_weighted_edgelist(dataset, nodetype=int)
    gt_path = "../dataset/real/" + network + "/labels.dat"
    ground_truth = load_ground_truth(gt_path)

    eps= estimate_eps_kneedle(G, similarity_func=similarity.Gen_wscan_similarity, mu=4, gamma=0.5)
    print("estimated eps =", eps)