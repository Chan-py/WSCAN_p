import argparse
from collections import defaultdict

import networkx as nx

def _id_cast(x):
    """노드 ID가 숫자 문자열이면 int로, 아니면 원문 str 유지."""
    try:
        return int(x)
    except Exception:
        return str(x)

def load_metadata_txt(meta_path):
    """metadata.txt: 'id  classname  gender' (공백/탭 구분). id→classname dict 반환."""
    id2class = {}
    with open(meta_path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"[metadata:{ln}] malformed: {line!r}")
            # 형식: id class [gender]
            i = _id_cast(parts[0])
            Ci = parts[1]
            id2class[i] = Ci
    return id2class

def read_gexf(path):
    """GEXF를 읽어 Graph와 함께 노드/엣지 속성을 유지."""
    G = nx.read_gexf(path)
    # 노드 ID 정규화 (int 가능하면 int로)
    if any(isinstance(n, str) for n in G.nodes()):
        mapping = {n: _id_cast(n) for n in list(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=True)
    return G

def collect_class_labels(graphs, meta_map):
    """GEXF 노드 속성(classname)과 metadata를 합쳐 id→classname 생성."""
    name2class = {}
    for G in graphs:
        if G is None:
            continue
        for n, d in G.nodes(data=True):
            c = d.get("classname")
            if c is not None and c != "":
                name2class[n] = str(c)
    # 메타데이터로 보강(없을 때만)
    for n, c in meta_map.items():
        name2class.setdefault(n, c)
    return name2class

def accumulate_edges(graphs, weight_attr="duration"):
    """
    여러 GEXF를 무방향으로 합산.
    weight_attr in {'duration','count'}.
    반환: edges dict[(u,v)] -> weight, nodes_seen set
    """
    edges = defaultdict(float)
    nodes_seen = set()
    for G in graphs:
        if G is None:
            continue
        nodes_seen |= set(G.nodes())
        for u, v, d in G.edges(data=True):
            if u == v:
                continue
            w = d.get(weight_attr)
            if w is None:
                # 혹시 'weight'로 들어온 경우 대비
                w = d.get("weight", 0.0)
            try:
                w = float(w)
            except Exception:
                continue
            a, b = (u, v) if u < v else (v, u)
            edges[(a, b)] += w
    return edges, nodes_seen

def write_outputs(edges, nodes_used, id2class_all, network_out, labels_out):
    # network.dat
    with open(network_out, "w", encoding="utf-8") as f:
        for (a, b), w in sorted(edges.items()):
            if a in nodes_used and b in nodes_used:
                # 정수로 떨어지면 int로, 아니면 float로
                if abs(w - int(w)) < 1e-9:
                    f.write(f"{a} {b} {int(w)}\n")
                else:
                    f.write(f"{a} {b} {w:.6g}\n")

    # labels.dat (그래프에 실제 등장한 노드만)
    # class → 정수 id 매핑
    classes = sorted({id2class_all[n] for n in nodes_used if n in id2class_all})
    class2id = {c: i for i, c in enumerate(classes)}  # 0..C-1

    with open(labels_out, "w", encoding="utf-8") as f:
        for n in sorted(nodes_used):
            c = id2class_all.get(n)
            if c is not None:
                f.write(f"{n} {class2id[c]}\n")

    # 보조 매핑 파일(사람이 보기 편함)
    with open("idmap_nodes.tsv", "w", encoding="utf-8") as f:
        for n in sorted(nodes_used):
            f.write(f"{n}\t{n}\n")  # 원본ID=정수ID라 자기 자신 매핑
    with open("idmap_classes.tsv", "w", encoding="utf-8") as f:
        for c, cid in sorted(class2id.items(), key=lambda x: x[1]):
            f.write(f"{cid}\t{c}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day1", required=True, help="sp_data_school_day_1_g.gexf")
    ap.add_argument("--day2", required=True, help="sp_data_school_day_2_g.gexf")
    ap.add_argument("--meta", required=True, help="metadata.txt (whitespace-separated)")
    ap.add_argument("--weight_attr", choices=["duration", "count"], default="duration",
                    help="엣지 가중치로 사용할 속성 (기본: duration)")
    ap.add_argument("--merge", choices=["both", "day1", "day2"], default="both",
                    help="두 날을 합산(both) 또는 특정 하루만 사용")
    ap.add_argument("--network_out", default="network.dat")
    ap.add_argument("--labels_out", default="labels.dat")
    args = ap.parse_args()

    G1 = read_gexf(args.day1)
    G2 = read_gexf(args.day2)
    graphs = {"both": [G1, G2], "day1": [G1], "day2": [G2]}[args.merge]

    meta_map = load_metadata_txt(args.meta)
    id2class = collect_class_labels(graphs, meta_map)
    edges, nodes_seen = accumulate_edges(graphs, weight_attr=args.weight_attr)

    write_outputs(edges, nodes_seen, id2class, args.network_out, args.labels_out)

    print(f"✅ nodes={len(nodes_seen)}  edges={len(edges)}  classes={len(set(id2class.values()))}")
    print(f"weight_attr={args.weight_attr}  merge={args.merge}")
    print(f"→ {args.network_out}, {args.labels_out}, idmap_nodes.tsv, idmap_classes.tsv")

if __name__ == "__main__":
    main()