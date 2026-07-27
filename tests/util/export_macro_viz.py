"""Macro door 배향 인터랙티브 HTML 용 데이터 export (JSON).

PYTHONHASHSEED=0 로 실행해야 클러스터링이 결정적이다:
    PYTHONHASHSEED=0 python tests/util/export_macro_viz.py
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from mr2s_module.cycle import FaceClusterPartition
from mr2s_module.domain import Edge, Graph
from tests.util.robbins_door_bruteforce import (
    SEED,
    build_delaunay,
    extract_cells_and_shared,
    prune_keep_biconnected,
)

OUT = "tests/util/macro_viz_data.json"


def main() -> None:
    import random

    raw, pts = build_delaunay(200, SEED)
    pruned = prune_keep_biconnected(raw, 0.4, SEED)
    cells, _shared = extract_cells_and_shared(pruned, pts)

    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])
    np.random.seed(SEED)
    random.seed(SEED)
    result = FaceClusterPartition(target_k=10).run(graph)

    # edge_key -> macro owners / door 방향
    key2macros: dict[tuple[int, int], set[int]] = {}
    key2door: dict[tuple[int, int], tuple[int, int]] = {}
    key2eid: dict[tuple[int, int], int] = {}
    for mi, sg in enumerate(result.sub_graphs):
        for e in sg.edges.values():
            k = tuple(sorted(e.endpoints()))
            key2macros.setdefault(k, set()).add(mi)
            key2eid[k] = e.id
            if e.directed:
                key2door[k] = e.vertices  # (tail, head)

    # 비어있지 않은 macro 만 재인덱싱
    used = sorted({m for ms in key2macros.values() for m in ms})
    remap = {old: new for new, old in enumerate(used)}
    n_macros = len(used)

    # macro 별 정점 수(동점 tie-break: 둘러싸인 섬 macro 가 더 작다).
    macro_size: dict[int, int] = {}
    for mi, sg in enumerate(result.sub_graphs):
        vs: set[int] = set()
        for e in sg.edges.values():
            a, b = e.endpoints()
            vs.add(a)
            vs.add(b)
        macro_size[mi] = len(vs)

    def _best(votes: Counter[int]) -> int:
        # 득표 최대 → 동점 시 정점 적은 macro(=enclosed 섬) 우선.
        best = max(votes.items(), key=lambda kv: (kv[1], -macro_size[kv[0]]))
        return remap[best[0]]

    def face_macro(cell: list[int]) -> int:
        # 1순위: internal(비-door) 간선이 가리키는 단일 macro 투표.
        votes: Counter[int] = Counter()
        for i in range(len(cell)):
            k = tuple(sorted((cell[i], cell[(i + 1) % len(cell)])))
            ms = key2macros.get(k, set())
            if len(ms) == 1:
                votes[next(iter(ms))] += 1
        if votes:
            return _best(votes)
        # 2순위(면 전 변이 door): 전 간선 owner 투표. macro0 에 둘러싸인
        # door-only 홑삼각형 macro 는 3-3 동점 → 작은 macro 로 tie-break.
        for i in range(len(cell)):
            k = tuple(sorted((cell[i], cell[(i + 1) % len(cell)])))
            for m in key2macros.get(k, ()):
                votes[m] += 1
        if votes:
            return _best(votes)
        return -1

    faces = [
        {"verts": [int(v) for v in c], "macro": face_macro(c)} for c in cells
    ]

    # doors: 전역 고유 edge (macro 2곳 공유 + directed)
    doors = []
    for k, th in sorted(key2door.items(), key=lambda kv: key2eid[kv[0]]):
        u, v = k
        doors.append(
            {
                "eid": key2eid[k],
                "u": int(u),
                "v": int(v),
                "tail": int(th[0]),
                "head": int(th[1]),
                "macros": sorted(remap[m] for m in key2macros[k]),
            }
        )

    # macros: 각 macro 의 간선(BFS 용). door 는 eid 로 전역 방향 참조, free 는 무방향.
    macros = []
    for old in used:
        sg = result.sub_graphs[old]
        edges = []
        verts: set[int] = set()
        for e in sg.edges.values():
            u, v = e.endpoints()
            verts.add(int(u))
            verts.add(int(v))
            k = tuple(sorted((u, v)))
            edges.append(
                {
                    "eid": e.id,
                    "u": int(u),
                    "v": int(v),
                    "door": k in key2door,
                }
            )
        macros.append(
            {"id": remap[old], "verts": sorted(verts), "edges": edges}
        )

    # 모든 간선(렌더용): free vs door 구분
    all_edges = []
    for u, v in pruned.edges():
        k = tuple(sorted((int(u), int(v))))
        all_edges.append(
            {"u": int(u), "v": int(v), "door": k in key2door, "eid": key2eid.get(k, -1)}
        )

    data = {
        "seed": SEED,
        "n_vertices": pruned.number_of_nodes(),
        "n_edges": pruned.number_of_edges(),
        "n_faces": len(cells),
        "n_macros": n_macros,
        "n_doors": len(doors),
        "verts": {int(i): [float(pts[i, 0]), float(pts[i, 1])] for i in range(len(pts))},
        "faces": faces,
        "edges": all_edges,
        "doors": doors,
        "macros": macros,
    }
    with open(OUT, "w") as f:
        json.dump(data, f)
    print(
        f"wrote {OUT}: V={data['n_vertices']} E={data['n_edges']} "
        f"faces={data['n_faces']} macros={n_macros} doors={len(doors)}"
    )


if __name__ == "__main__":
    main()
