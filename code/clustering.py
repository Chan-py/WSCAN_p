from collections import deque

def run(G, similarity_func, eps=0.5, mu=2, gamma=1):
    # # For Debugging
    # for u in G.nodes():
    #     for v in G.neighbors(u):
    #         print(u, v, similarity_func(G, u, v, gamma))


    cores = {u for u in G.nodes()
            if sum(is_eps_neighbor(G, u, v, eps, similarity_func, gamma) for v in G.neighbors(u)) >= mu}
    # print(cores)
    
    # Cluster expansion
    clusters = {}
    visited = set()
    cluster_id = 0
    for u in cores:
        if u in visited:
            continue
        cluster_id += 1
        clusters[cluster_id] = set()
        queue = deque([u])
        visited.add(u)
        while queue:
            x = queue.popleft()
            clusters[cluster_id].add(x)
            for y in G.neighbors(x):
                if y not in visited and y in cores and is_eps_neighbor(G, x, y, eps, similarity_func, gamma):
                    visited.add(y)
                    queue.append(y)
        # Include directly reachable non-core
        for member in list(clusters[cluster_id]):
            for y in G.neighbors(member):
                if y not in visited and is_eps_neighbor(G, member, y, eps, similarity_func, gamma):
                    clusters[cluster_id].add(y)
                    visited.add(y)
                    
    # Classify hubs and outliers
    all_clustered = set().union(*clusters.values()) if clusters else set()
    hubs, outliers = set(), set()
    for u in G.nodes():
        if u in all_clustered:
            continue
        connected = {cid for cid, members in clusters.items()
                        if any(v in members for v in G.neighbors(u))}
        if len(connected) >= 2:
            hubs.add(u)
        else:
            outliers.add(u)
    return clusters, hubs, outliers

def is_eps_neighbor(G, u, v, eps, similarity_func, gamma):
    return similarity_func(G, u, v, gamma) >= eps