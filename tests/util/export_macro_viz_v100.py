"""V=100 macro 그래프 시각화용 데이터 export (JSON).

실제 파이프라인(Delaunay → biconnected prune → FaceClusterPartition)을 그대로
돌려 macro/door/geometry 를 뽑는다. export_macro_viz.py 의 V=100 판.

    PYTHONHASHSEED=0 python tests/util/export_macro_viz_v100.py
"""

from __future__ import annotations

import json
import random
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

N = 100
OUT = "tests/util/macro_viz_v100.json"


def main() -> None:
    raw, pts = build_delaunay(N, SEED)
    pruned = prune_keep_biconnected(raw, 0.4, SEED)
    cells, _shared = extract_cells_and_shared(pruned, pts)

    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])
    np.random.seed(SEED)
    random.seed(SEED)
    result = FaceClusterPartition(target_k=10).run(graph)

    key2macros: dict[tuple[int, int], set[int]] = {}
    key2door: dict[tuple[int, int], tuple[int, int]] = {}
    key2eid: dict[tuple[int, int], int] = {}
    for mi, sg in enumerate(result.sub_graphs):
        for e in sg.edges.values():
            k = tuple(sorted(e.endpoints()))
            key2macros.setdefault(k, set()).add(mi)
            key2eid[k] = e.id
            if e.directed:
                key2door[k] = e.vertices

    used = sorted({m for ms in key2macros.values() for m in ms})
    remap = {old: new for new, old in enumerate(used)}
    n_macros = len(used)

    macro_size: dict[int, int] = {}
    for mi, sg in enumerate(result.sub_graphs):
        vs: set[int] = set()
        for e in sg.edges.values():
            a, b = e.endpoints()
            vs.add(a)
            vs.add(b)
        macro_size[mi] = len(vs)

    def _best(votes: Counter[int]) -> int:
        best = max(votes.items(), key=lambda kv: (kv[1], -macro_size[kv[0]]))
        return remap[best[0]]

    def face_macro(cell: list[int]) -> int:
        votes: Counter[int] = Counter()
        for i in range(len(cell)):
            k = tuple(sorted((cell[i], cell[(i + 1) % len(cell)])))
            ms = key2macros.get(k, set())
            if len(ms) == 1:
                votes[next(iter(ms))] += 1
        if votes:
            return _best(votes)
        for i in range(len(cell)):
            k = tuple(sorted((cell[i], cell[(i + 1) % len(cell)])))
            for m in key2macros.get(k, ()):
                votes[m] += 1
        if votes:
            return _best(votes)
        return -1

    faces = [{"verts": [int(v) for v in c], "macro": face_macro(c)} for c in cells]

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
            edges.append({"eid": e.id, "u": int(u), "v": int(v), "door": k in key2door})
        macros.append({"id": remap[old], "verts": sorted(verts), "edges": edges})

    all_edges = []
    for u, v in pruned.edges():
        k = tuple(sorted((int(u), int(v))))
        all_edges.append(
            {"u": int(u), "v": int(v), "door": k in key2door, "eid": key2eid.get(k, -1)}
        )

    # 전체 그래프 사이클 차원 = 간선 - 정점 + 1 (연결 그래프)
    cycle_dim = pruned.number_of_edges() - pruned.number_of_nodes() + 1

    # door 부분그래프(문 간선만)의 사이클 차원 = E - V + C.
    # 이것이 door 방향의 실질 자유도(닫힌 door 고리 수)다.
    import networkx as nx

    door_g = nx.Graph()
    for d in doors:
        door_g.add_edge(d["u"], d["v"])
    n_comp = nx.number_connected_components(door_g) if door_g.number_of_nodes() else 0
    door_cycle_dim = (
        door_g.number_of_edges() - door_g.number_of_nodes() + n_comp
        if door_g.number_of_nodes()
        else 0
    )

    data = {
        "seed": SEED,
        "n_vertices": pruned.number_of_nodes(),
        "n_edges": pruned.number_of_edges(),
        "n_faces": len(cells),
        "n_macros": n_macros,
        "n_doors": len(doors),
        "cycle_dim": cycle_dim,
        "door_cycle_dim": door_cycle_dim,
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
        f"faces={data['n_faces']} macros={n_macros} doors={len(doors)} "
        f"cycle_dim={cycle_dim} door_cycle_dim={door_cycle_dim}"
    )


if __name__ == "__main__":
    main()
