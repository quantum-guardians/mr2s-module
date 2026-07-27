"""door 배향 선택이 QUBO 도달 최적 품질(APSP)을 바꾸는가?

door 는 QUBO 변수가 아니라 미리 못박는 값(2색칠 → 한 valid 배향). valid door
배향은 여러 개다(part1). 각 valid door 배향마다 '비-door 간선을 자유 배향해
얻는 최소 APSP(=QUBO 도달 최적)'를 구해, door 선택이 품질을 흔드는지 본다.

    - door 선택이 최적 APSP 를 바꾼다 → door 배향은 품질 레버
    - 안 바꾼다(전역 반전쌍 빼고 동일) → DnC 일관성용일 뿐, 품질 무관

    PYTHONHASHSEED=0 python tests/util/door_orientation_quality.py [V] [K]

실험 전용.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from itertools import product

import networkx as nx
import numpy as np

from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.robbins_door_bruteforce import (
    SEED,
    build_delaunay,
    prune_keep_biconnected,
)

N_VERTICES = int(sys.argv[1]) if len(sys.argv) > 1 else 26
REMOVE_RATIO = 0.4
TARGET_K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NONDOOR_CAP = 16
DOOR_CAP = 12


def apsp_sum(edges, verts):
    g = nx.DiGraph()
    g.add_nodes_from(verts)
    g.add_edges_from(edges)
    if not nx.is_strongly_connected(g):
        return None
    total = 0.0
    for _s, dist in nx.all_pairs_shortest_path_length(g):
        total += sum(dist.values())
    return total


def main() -> None:
    raw, _pts = build_delaunay(N_VERTICES, SEED)
    pruned = prune_keep_biconnected(raw, REMOVE_RATIO, SEED)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])
    np.random.seed(SEED)
    random.seed(SEED)
    do = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)

    door_eids = {
        e.id for sg in do.sub_graphs for e in sg.edges.values() if e.directed
    }
    print(f"V={pruned.number_of_nodes()} E={pruned.number_of_edges()} "
          f"door={len(door_eids)}")
    print()
    print(f"{'macro':>5} {'|V|':>4} {'door':>4} {'nondoor':>7} "
          f"{'valid_door':>10} {'apsp분포(최적 over free)':>28}")

    for mi, sg in enumerate(do.sub_graphs):
        if not sg.edges:
            continue
        verts: set[int] = set()
        door_e: list[tuple[int, int]] = []
        nondoor_e: list[tuple[int, int]] = []
        for e in sg.edges.values():
            u, v = e.endpoints()
            verts.add(u)
            verts.add(v)
            (door_e if e.id in door_eids else nondoor_e).append((u, v))
        vlist = sorted(verts)
        if len(door_e) > DOOR_CAP or len(nondoor_e) > NONDOOR_CAP:
            print(f"{mi:>5} {len(vlist):>4} {len(door_e):>4} {len(nondoor_e):>7} "
                  f"{'skip':>10}")
            continue

        # door 배향별: 비-door 자유배향 중 강연결 최소 APSP (= QUBO 도달 최적)
        best_per_door: dict[tuple[int, ...], float] = {}
        for dbits in product((0, 1), repeat=len(door_e)):
            fixed = [
                door_e[i] if dbits[i] == 0 else (door_e[i][1], door_e[i][0])
                for i in range(len(door_e))
            ]
            best = None
            for nbits in product((0, 1), repeat=len(nondoor_e)):
                oriented = list(fixed) + [
                    nondoor_e[j] if nbits[j] == 0 else (nondoor_e[j][1], nondoor_e[j][0])
                    for j in range(len(nondoor_e))
                ]
                s = apsp_sum(oriented, vlist)
                if s is not None and (best is None or s < best):
                    best = s
            if best is not None:
                best_per_door[dbits] = best

        if not best_per_door:
            print(f"{mi:>5} {len(vlist):>4} {len(door_e):>4} {len(nondoor_e):>7} "
                  f"{'0(강연결불가)':>10}")
            continue
        vals = sorted(best_per_door.values())
        dist = Counter(vals)
        summary = f"min={vals[0]:.0f} max={vals[-1]:.0f} 종류{len(dist)} {dict(dist)}"
        print(f"{mi:>5} {len(vlist):>4} {len(door_e):>4} {len(nondoor_e):>7} "
              f"{len(best_per_door):>10} {summary}")


if __name__ == "__main__":
    main()
