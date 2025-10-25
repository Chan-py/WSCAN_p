import argparse
from collections import defaultdict

def load_metadata(meta_path):
    """
    metadata.txt: 'i  Di' (공백/탭 구분)
    return:
      id2dept: {id:int or str -> dept_code:str}
      dept2id: {dept_code -> int}  # 0..C-1
    """
    id2dept = {}
    with open(meta_path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"[metadata:{ln}] malformed: {line!r}")
            i = parts[0]
            # ID가 숫자면 int로, 아니면 문자열로 보존
            try:
                i = int(i)
            except ValueError:
                pass
            dept = parts[1]
            id2dept[i] = dept
    depts = sorted(set(id2dept.values()))
    dept2id = {d: k for k, d in enumerate(depts)}  # 0..C-1
    return id2dept, dept2id

def build_weighted_network(contacts_path, unit="seconds"):
    """
    tij_InVS.dat: 't  i  j' (공백/탭)
    각 라인은 20초 창에서의 활성 접촉 → (i,j) weight에 +add
    unit: 'seconds' → +20, 'minutes' → +1/3, 'windows' → +1
    return:
      edges: {(u,v): weight}  # 무방향, u<v
      nodes_seen: set of ids
    """
    assert unit in {"seconds", "minutes", "windows"}
    add = 20.0 if unit == "seconds" else (20.0/60.0 if unit == "minutes" else 1.0)

    edges = defaultdict(float)
    nodes_seen = set()

    with open(contacts_path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                # 헤더/잡음 라인 스킵
                continue
            # 시간 t는 쓰지 않음(합산만 함)
            try:
                _ = float(parts[0])
                i = parts[1]; j = parts[2]
            except Exception:
                # 헤더일 가능성
                continue
            # ID 숫자화(가능하면)
            for var in ("i", "j"):
                try:
                    if var == "i":
                        i = int(i)
                    else:
                        j = int(j)
                except ValueError:
                    pass

            if i == j:
                continue
            u, v = (i, j) if (str(i) < str(j)) else (j, i)
            edges[(u, v)] += add
            nodes_seen.update([i, j])

    return edges, nodes_seen

def write_outputs(edges, id2dept, dept2id, nodes_used, network_out, labels_out):
    # network.dat : u v w (무방향)
    with open(network_out, "w", encoding="utf-8") as f:
        for (u, v), w in sorted(edges.items()):
            if u in nodes_used and v in nodes_used:
                if abs(w - int(w)) < 1e-9:
                    f.write(f"{u} {v} {int(w)}\n")
                else:
                    f.write(f"{u} {v} {w:.6g}\n")

    # labels.dat : id dept_id  (라벨 있는 노드만)
    with open(labels_out, "w", encoding="utf-8") as f:
        for n in sorted(nodes_used, key=str):
            dept = id2dept.get(n)
            if dept is not None:
                f.write(f"{n} {dept2id[dept]}\n")

    # 참고 매핑(사람이 보기 편한 파일)
    with open("idmap_departments.tsv", "w", encoding="utf-8") as f:
        for d, did in sorted(dept2id.items(), key=lambda x: x[1]):
            f.write(f"{did}\t{d}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contacts", required=True, help="tij_InVS.dat (t i j)")
    ap.add_argument("--meta", required=True, help="metadata.txt (i Di)")
    ap.add_argument("--unit", choices=["seconds", "minutes", "windows"], default="seconds",
                    help="가중치 단위: 초/분 누적 또는 20초 창 개수")
    ap.add_argument("--network_out", default="network.dat")
    ap.add_argument("--labels_out", default="labels.dat")
    ap.add_argument("--only_labeled", action="store_true",
                    help="라벨 없는 노드는 네트워크에서 제외(평가 집합과 일치시키려면 권장)")
    args = ap.parse_args()

    id2dept, dept2id = load_metadata(args.meta)
    edges, nodes_seen = build_weighted_network(args.contacts, unit=args.unit)

    nodes_used = nodes_seen if not args.only_labeled else (nodes_seen & set(id2dept.keys()))
    # 라벨 없는 노드 제거 시, 엣지는 write_outputs에서 nodes_used로 필터링됨

    write_outputs(edges, id2dept, dept2id, nodes_used, args.network_out, args.labels_out)

    print(f"✅ nodes(used)={len(nodes_used)}  edges={sum(1 for (u,v) in edges if u in nodes_used and v in nodes_used)}  depts={len(dept2id)}")
    print(f"unit={args.unit}  only_labeled={args.only_labeled}")
    print(f"→ {args.network_out}, {args.labels_out}, idmap_departments.tsv")

if __name__ == "__main__":
    main()
