from collections import deque
import time
from parallel import run_parallel_cores, run_parallel_cores_include_me

def run(G, similarity_func, eps=0.5, mu=2, gamma=1, parallel=False, workers=None):
    # # For Debugging
    # for u in G.nodes():
    #     for v in G.neighbors(u):
    #         print(u, v, similarity_func(G, u, v, gamma))

    if not parallel:
        n_start = time.time()
        cores = {u for u in G.nodes()
                if sum(is_eps_neighbor(G, u, v, eps, similarity_func, gamma) for v in G.neighbors(u)) >= mu}
        similarity_calculating_time = time.time() - n_start
    else:
        cores, similarity_calculating_time = run_parallel_cores(G, similarity_func, eps, mu, gamma, workers)
    
    label = {}
    for u in G.nodes():
        label[u] = -1  # unclassified
    
    visited = {}
    for u in cores:
        visited[u] = False
    
    # Cluster expansion
    clusters = {}
    hubs, outliers = set(), set()
    
    cluster_id = 0
    for u in cores:
        if visited[u]:
            continue
        cluster_id += 1
        clusters[cluster_id] = set()
        # Initialize Queue
        queue = deque([u])
        visited[u] = True
        
        label[u] = cluster_id
        clusters[cluster_id].add(u)
        
        while queue:
            x = queue.popleft()
            for v in G.neighbors(x):
                if is_eps_neighbor(G, x, v, eps, similarity_func, gamma) and label[v] == -1:
                    label[v] = cluster_id
                    clusters[cluster_id].add(v)
                    if v in cores and not visited[v]:
                        queue.append(v)
                        visited[v] = True
                    
    # Classify hubs and outliers
    for u in G.nodes():
        if label[u] != -1:
            continue
        connected = {label[v] for v in G.neighbors(u) if label[v] != -1}
        
        if len(connected) >= 2:
            hubs.add(u)
        else:
            outliers.add(u)
    return clusters, hubs, outliers, similarity_calculating_time


def run_include_me(G, similarity_func, eps=0.5, mu=2, gamma=1, parallel=False, workers=None):
    # # For Debugging
    # for u in G.nodes():
    #     for v in G.neighbors(u):
    #         print(u, v, similarity_func(G, u, v, gamma))

    if not parallel:
        n_start = time.time()
        cores = {u for u in G.nodes()
                if sum(is_eps_neighbor(G, u, v, eps, similarity_func, gamma) for v in G.neighbors(u)) + 1 >= mu}
        similarity_calculating_time = time.time() - n_start
    else:
        cores, similarity_calculating_time = run_parallel_cores_include_me(G, similarity_func, eps, mu, gamma, workers)
    
    label = {}
    for u in G.nodes():
        label[u] = -1  # unclassified
    
    visited = {}
    for u in cores:
        visited[u] = False
    
    # Cluster expansion
    clusters = {}
    hubs, outliers = set(), set()
    
    cluster_id = 0
    for u in cores:
        if visited[u]:
            continue
        cluster_id += 1
        clusters[cluster_id] = set()
        # Initialize Queue
        queue = deque([u])
        visited[u] = True
        
        label[u] = cluster_id
        clusters[cluster_id].add(u)
        
        while queue:
            x = queue.popleft()
            for v in G.neighbors(x):
                if is_eps_neighbor(G, x, v, eps, similarity_func, gamma) and label[v] == -1:
                    label[v] = cluster_id
                    clusters[cluster_id].add(v)
                    if v in cores and not visited[v]:
                        queue.append(v)
                        visited[v] = True
                    
    # Classify hubs and outliers
    for u in G.nodes():
        if label[u] != -1:
            continue
        connected = {label[v] for v in G.neighbors(u) if label[v] != -1}
        
        if len(connected) >= 2:
            hubs.add(u)
        else:
            outliers.add(u)
    return clusters, hubs, outliers, similarity_calculating_time


def is_eps_neighbor(G, u, v, eps, similarity_func, gamma):
    return similarity_func(G, u, v, gamma) >= eps