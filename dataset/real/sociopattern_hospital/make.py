import argparse
from collections import defaultdict

def parse_contacts(path, unit="seconds"):
    """
    읽기: 한 줄 't i j Si Sj'
      - t: 정수(초). 각 줄은 20초 길이의 활성 구간의 끝 시각.
      - i, j: 익명 ID (정수 권장)
      - Si, Sj: 상태 라벨 문자열 {NUR, PAT, MED, ADM}

    반환:
      edges[(u,v)] -> weight  (무방향, 누적)
      id2status[id] -> status (문자열)
    """
    assert unit in {"seconds", "minutes", "windows"}
    add = 20.0 if unit == "seconds" else (20.0/60.0 if unit == "minutes" else 1.0)

    edges = defaultdict(float)
    id2status = {}

    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                # 헤더/잡음 라인 방지
                continue
            # t, i, j
            try:
                # 첫 토큰이 숫자가 아니면 헤더일 수 있음 → 스킵
                _ = float(parts[0])
                i = int(parts[1]); j = int(parts[2])
            except Exception:
                continue
            Si = parts[3].strip()
            Sj = parts[4].strip()

            # 라벨 누적 (충돌 체크)
            for node, st in ((i, Si), (j, Sj)):
                prev = id2status.get(node)
                if prev is None:
                    id2status[node] = st
                elif prev != st:
                    # 실제 데이터에서는 거의 발생하지 않지만,
                    # 만약 있으면 첫 라벨을 유지(원하면 assert로 강제 검사해도 됨)
                    pass

            if i == j:  # 자기 자신 무시
                continue
            u, v = (i, j) if i < j else (j, i)
            edges[(u, v)] += add

    return edges, id2status

def write_outputs(edges, id2status, network_out, labels_out):
    # network.dat
    with open(network_out, "w", encoding="utf-8") as f:
        for (u, v), w in sorted(edges.items()):
            # 정수로 떨어지면 int로 출력
            if abs(w - int(w)) < 1e-9:
                f.write(f"{u} {v} {int(w)}\n")
            else:
                f.write(f"{u} {v} {w:.6g}\n")

    # status → 정수 ID 매핑(안정적 출력을 위해 사전식 정렬)
    statuses = sorted(set(id2status.values()))
    status2id = {s: k for k, s in enumerate(statuses)}  # 0..C-1

    # labels.dat
    with open(labels_out, "w", encoding="utf-8") as f:
        for node in sorted(id2status.keys()):
            f.write(f"{node} {status2id[id2status[node]]}\n")

    # 보조 매핑 파일
    with open("idmap_statuses.tsv", "w", encoding="utf-8") as f:
        for s, sid in sorted(status2id.items(), key=lambda x: x[1]):
            f.write(f"{sid}\t{s}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contacts", required=True, help="temporal contacts file (t i j Si Sj)")
    ap.add_argument("--network_out", default="network.dat")
    ap.add_argument("--labels_out", default="labels.dat")
    ap.add_argument("--unit", choices=["seconds", "minutes", "windows"], default="seconds",
                    help="가중치 단위: 초/분 누적 또는 창 개수(windows)")
    args = ap.parse_args()

    edges, id2status = parse_contacts(args.contacts, unit=args.unit)
    write_outputs(edges, id2status, args.network_out, args.labels_out)

    print(f"✅ nodes={len(id2status)}  edges={len(edges)}  statuses={len(set(id2status.values()))}")
    print(f"unit={args.unit} → {args.network_out}, {args.labels_out}, idmap_statuses.tsv 생성")

if __name__ == "__main__":
    main()
