import math

def scan_similarity(G, u, v, gamma):
    neigh_u = set(G.neighbors(u)) | {u}
    neigh_v = set(G.neighbors(v)) | {v}
    common = neigh_u & neigh_v
    if not common:
        return 0.0
    return len(common) / math.sqrt(len(neigh_u) * len(neigh_v))

def wscan_similarity(G, u, v, gamma):
    """
    WSCAN similarity:
      sim(u,v) = scan_similarity(u,v) * w(u,v)
    """
    # direct edge weight
    w_uv = G[u][v]['weight'] if G.has_edge(u, v) else 0.0
    return scan_similarity(G, u, v, gamma) * w_uv

def cosine_similarity(G, u, v, gamma):
    """
    Cosine Similarity:
    s_cos'(u,v) = (
        w(u,v)^2 + sum_{x in N(u)∩N(v)} w(u,x)*w(v,x)
    ) / (
        sqrt(sum_{x in N(u)} w(u,x)^2)
        * sqrt(sum_{y in N(v)} w(v,y)^2)
    )
    """
    # 이웃 집합
    neigh_u = set(G.neighbors(u))
    neigh_v = set(G.neighbors(v))
    common = neigh_u & neigh_v

    # direct edge weight
    w_uv = G[u][v]['weight'] if G.has_edge(u, v) else 0.0
    # 분자: direct 연결 기여 (w_uv^2) + 공통 이웃 기여
    numer = w_uv**2 + sum(
        G[u][x]['weight'] * G[v][x]['weight']
        for x in common
    )

    # 분모: 각 노드의 L2 norm
    norm_u = math.sqrt(sum(data['weight']**2 for _, _, data in G.edges(u, data=True)))
    norm_v = math.sqrt(sum(data['weight']**2 for _, _, data in G.edges(v, data=True)))
    if norm_u == 0 or norm_v == 0:
        return 0.0

    return numer / (norm_u * norm_v)

def Gen_wscan_similarity(G, u, v, gamma):
    """
    Generalized wscan similarity:
    s_str'(u,v) = w(u,v) + gamma * sum_{x in N(u)∩N(v)} min(w(u,x), w(v,x))
                  ---------------------------------------------------
                  sqrt( (sum_{x in N(u)} w(u,x)) * (sum_{y in N(v)} w(v,y)) )
    """
    # Common Neighbors
    neigh_u = set(G.neighbors(u))
    neigh_v = set(G.neighbors(v))
    common = neigh_u & neigh_v

    # direct edge weight
    w_uv = G[u][v]['weight'] if G.has_edge(u, v) else 0.0

    # 공통 이웃에 대한 min-weight 합
    common_sum = sum(
        min(G[u][x]['weight'], G[v][x]['weight'])
        for x in common
    )

    # 분자: direct + common
    numer = w_uv + gamma * common_sum

    # 분모: 각 노드의 total strength
    su = sum(data['weight'] for _, _, data in G.edges(u, data=True))
    sv = sum(data['weight'] for _, _, data in G.edges(v, data=True))
    if su == 0 or sv == 0:
        return 0.0

    return numer / math.sqrt(su * sv)


# 번외 similarity functions
def weighted_jaccard_similarity(G, u, v, gamma):
    """
    Weighted Jaccard Similarity (with direct edge term and intersection only):
    s_J(u,v) = w(u,v)
               + sum_{x in N(u) ∩ N(v)} min(w(u,x), w(v,x))
               ---------------------------------------------------
               sum_{x in N(u) ∪ N(v)} max(w(u,x), w(v,x))
    """
    neigh_u = set(G.neighbors(u))
    neigh_v = set(G.neighbors(v))
    common_neigh = neigh_u & neigh_v
    all_neigh = neigh_u | neigh_v

    # numerator: direct edge + shared‐neighbor min‐weights
    w_uv = G[u][v]['weight'] if G.has_edge(u, v) else 0.0
    numer = w_uv
    for x in common_neigh:
        wu = G[u][x]['weight']
        wv = G[v][x]['weight']
        numer += min(wu, wv)

    # denominator: union max‐weights
    denom = 0.0
    for x in all_neigh:
        wu = G[u][x]['weight'] if x in neigh_u else 0.0
        wv = G[v][x]['weight'] if x in neigh_v else 0.0
        denom += max(wu, wv)

    return numer / denom if denom > 0 else 0.0