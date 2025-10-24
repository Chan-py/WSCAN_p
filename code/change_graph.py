import random
import networkx as nx

def perturb_edge_weights(G: nx.Graph, delta_percent: float, edge_percent: float, weight_method) -> None:
    m = G.number_of_edges()
    if m == 0:
        return

    try:
        if weight_method == "max":
            w_representative = max(data.get("weight", 1.0) for _, _, data in G.edges(data=True))
        elif weight_method == "avg":
            w_representative = sum(data.get("weight", 1.0) for _, _, data in G.edges(data=True)) / m
    except ValueError:
        print("error checking edge weight")
        return

    delta = delta_percent * w_representative
    k = int(round(edge_percent * m))
    k = max(0, min(m, k))
    print(delta, k)

    if k == 0 or delta == 0:
        return

    edges = list(G.edges())
    chosen_edges = random.sample(edges, k)

    cnt = [0, 0]
    for u, v in chosen_edges:
        w = G[u][v].get("weight", 1.0)
        
        change = 0.0
        while change == 0.0:
            change = random.uniform(-delta, delta)
        
        new_weight = w + change
        
        if new_weight <= 0:
            G[u][v]["weight"] = w * 0.01
            cnt[0] += 1
        else:
            G[u][v]["weight"] = new_weight
            cnt[1] += 1
    print(f"Total edges decreased to near zero: {cnt[0]}, increased/decreased normally: {cnt[1]}")