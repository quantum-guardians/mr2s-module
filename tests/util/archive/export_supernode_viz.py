"""Supernode 축약 뷰 데이터 export.

각 macro 에서 free 내부간선으로 정점을 축약(supernode)하고, macro 간 door 를
supernode 사이 directed arc 로 남긴 본질 구조를 내보낸다. 이 구조 위에서
'강연결 배향 개수 = (대체로) 2^(cycle rank)' 를 눈으로 확인한다.

    PYTHONHASHSEED=0 python tests/util/export_supernode_viz.py
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np

from mr2s_module.cycle import FaceClusterPartition
from mr2s_module.domain import Edge, Graph
from tests.util.robbins_door_bruteforce import (
    SEED,
    build_delaunay,
    prune_keep_biconnected,
)

OUT = "tests/util/supernode_viz_data.json"


class DSU:
    def __init__(self, verts):
        self.p = {v: v for v in verts}

    def find(self, x):
        r = x
        while self.p[r] != r:
            r = self.p[r]
        self.p[x] = r
        return r

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main() -> None:
    import random

    raw, pts = build_delaunay(200, SEED)
    pruned = prune_keep_biconnected(raw, 0.4, SEED)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])
    np.random.seed(SEED)
    random.seed(SEED)
    result = FaceClusterPartition(target_k=10).run(graph)

    canon = {}
    for sg in result.sub_graphs:
        for e in sg.edges.values():
            if e.directed:
                canon[e.id] = e.vertices  # (tail, head)

    used = [mi for mi, sg in enumerate(result.sub_graphs) if len(sg.edges) > 0]
    macros = []
    for new_id, mi in enumerate(used):
        sg = result.sub_graphs[mi]
        verts = set()
        for e in sg.edges.values():
            u, v = e.endpoints()
            verts.add(u)
            verts.add(v)
        dsu = DSU(verts)
        for e in sg.edges.values():
            if not e.directed:  # free 내부간선으로 축약
                u, v = e.endpoints()
                dsu.union(u, v)
        roots = sorted({dsu.find(v) for v in verts})
        sid = {r: i for i, r in enumerate(roots)}
        # supernode 위치 = 포함 정점 centroid
        members = {i: [] for i in range(len(roots))}
        for v in verts:
            members[sid[dsu.find(v)]].append(v)
        supernodes = []
        for i in range(len(roots)):
            mv = members[i]
            c = np.mean([pts[v] for v in mv], axis=0)
            supernodes.append({"pos": [float(c[0]), float(c[1])], "size": len(mv)})
        # cross-door arc (supernode 다른 경우만)
        arcs = []
        for e in sg.edges.values():
            if e.directed:
                t, h = canon[e.id]
                sa, sb = sid[dsu.find(t)], sid[dsu.find(h)]
                if sa != sb:
                    arcs.append({"eid": e.id, "a": sa, "b": sb})
        n_sup = len(roots)
        n_arc = len(arcs)

        # cactus 판정: supernode 그래프의 biconnected block 중 |E|>|V| 인
        # 블록(= 사이클 여러 개가 간선/정점 공유)이 있으면 non-cactus.
        sn_graph = nx.MultiGraph()
        sn_graph.add_nodes_from(range(n_sup))
        for j, a in enumerate(arcs):
            sn_graph.add_edge(a["a"], a["b"], key=j)
        complex_edges = set()  # non-cactus 블록에 속한 arc 인덱스
        for block in nx.biconnected_component_edges(sn_graph):
            block = list(block)
            nodes = {x for e in block for x in (e[0], e[1])}
            if len(block) > len(nodes):  # |E|>|V| → 단일 사이클 초과
                for e in block:
                    complex_edges.add(e[2])  # edge key = arc index
        for j, a in enumerate(arcs):
            a["complex"] = j in complex_edges
        n_complex = sum(1 for a in arcs if a["complex"])
        macros.append({
            "id": new_id,
            "n_sup": n_sup,
            "supernodes": supernodes,
            "arcs": arcs,
            "cycle_rank": n_arc - n_sup + 1,
            "is_cactus": n_complex == 0,
        })

    data = {"seed": SEED, "macros": macros}
    with open(OUT, "w") as f:
        json.dump(data, f)
    print(f"wrote {OUT}")
    for m in macros:
        print(
            f"macro {m['id']}: sup {m['n_sup']}, arc {len(m['arcs'])}, "
            f"cycle_rank {m['cycle_rank']}, cactus={m['is_cactus']}"
        )


if __name__ == "__main__":
    main()
