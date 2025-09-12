import networkx as nx

# 그래프 불러오기
G = nx.karate_club_graph()

# 1. edge list 저장 (network.dat)
with open("network.dat", "w") as f:
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)  # weight 없으면 기본값 1
        # 1-based indexing으로 저장 (논문 원본과 맞추려면)
        f.write(f"{u+1} {v+1} {w}\n")

# 2. labels 저장 (labels.dat)
# ground truth: club 속성 ("Mr. Hi" → 0, "Officer" → 1 로 매핑)
club_to_label = {"Mr. Hi": 0, "Officer": 1}

with open("labels.dat", "w") as f:
    for node, data in G.nodes(data=True):
        label = club_to_label[data["club"]]
        # 1-based indexing 사용
        f.write(f"{node+1} {label}\n")

print("✅ Saved network.dat and labels.dat")