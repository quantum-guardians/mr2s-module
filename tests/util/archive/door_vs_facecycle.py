"""door-only vs face-cycle 강연결 해공간·품질(APSP) 비교 실험.

두 방식 모두 door(owner>=2 공유경계)는 2-coloring 이 정한 canonical 방향으로
고정한다. 차이는 '외곽경계(owner==1)' 간선:
    - face-cycle: 외곽도 고정  → free = 내부간선만
    - door-only : 외곽은 free  → free = 내부 + 외곽
따라서 S_do ⊇ S_fc (door-only 해공간이 외곽 자유도만큼 더 큼). 질문: door-only
의 더 큰 강연결 해공간이 face-cycle 이 못 뽑는 '더 낮은 APSP(=더 좋은 품질)'
배향을 품는가?

macro 별로 door 를 canonical 로 고정하고 비-door 간선 배향을 전수 열거해:
    S_do = 강연결 배향 전부
    S_fc = 그 중 외곽=canonical 인 것
각 배향의 APSP 합(방향 최단경로 총합, 낮을수록 좋음) 최소를 비교한다.

    PYTHONHASHSEED=0 python tests/util/door_vs_facecycle.py

실험 전용 — 프로덕션에 두지 않는다.
"""

from __future__ import annotations

import random
from itertools import product

import networkx as nx
import numpy as np

from mr2s_module.cycle import FaceClusterPartition
from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.robbins_door_bruteforce import (
    SEED,
    build_delaunay,
    prune_keep_biconnected,
)

import sys

N_VERTICES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REMOVE_RATIO = 0.4
TARGET_K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ENUM_CAP = 22  # 비-door 간선 수 상한(2^n 열거)


def directed_eids_and_dirs(result) -> dict[int, tuple[int, int]]:
    """result 의 모든 directed 간선 → {eid: (tail, head)}."""
    out: dict[int, tuple[int, int]] = {}
    for sg in result.sub_graphs:
        for e in sg.edges.values():
            if e.directed:
                out[e.id] = e.vertices
    return out


def apsp_sum(edges: list[tuple[int, int]], verts: list[int]) -> float | None:
    """방향 그래프 강연결이면 모든 순서쌍 최단경로(홉) 합, 아니면 None."""
    g = nx.DiGraph()
    g.add_nodes_from(verts)
    g.add_edges_from(edges)
    if not nx.is_strongly_connected(g):
        return None
    total = 0.0
    for _src, dist in nx.all_pairs_shortest_path_length(g):
        total += sum(dist.values())
    return total


def main() -> None:
    raw, _pts = build_delaunay(N_VERTICES, SEED)
    pruned = prune_keep_biconnected(raw, REMOVE_RATIO, SEED)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])

    np.random.seed(SEED)
    random.seed(SEED)
    fc = FaceClusterPartition(target_k=TARGET_K).run(graph)  # face-cycle: 전 경계 고정
    np.random.seed(SEED)
    random.seed(SEED)
    do = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)  # door 만 고정

    fc_dir = directed_eids_and_dirs(fc)   # door + 외곽
    do_dir = directed_eids_and_dirs(do)   # door 만
    door_eids = set(do_dir)
    outer_eids = set(fc_dir) - door_eids  # face-cycle 이 추가로 고정하는 외곽

    print(f"V={pruned.number_of_nodes()} E={pruned.number_of_edges()} "
          f"door={len(door_eids)} outer={len(outer_eids)}")
    print()
    print(f"{'macro':>5} {'|V|':>4} {'door':>4} {'outer':>5} {'intern':>6} "
          f"{'|S_do|':>8} {'|S_fc|':>8} {'apsp_do':>8} {'apsp_fc':>8} {'이득':>5}")

    for mi, sg in enumerate(fc.sub_graphs):
        if not sg.edges:
            continue
        verts: set[int] = set()
        door_e: list[tuple[int, int, int]] = []   # (eid,u,v)
        outer_e: list[tuple[int, int, int]] = []
        intern_e: list[tuple[int, int, int]] = []
        for e in sg.edges.values():
            u, v = e.endpoints()
            verts.add(u)
            verts.add(v)
            if e.id in door_eids:
                door_e.append((e.id, u, v))
            elif e.id in outer_eids:
                outer_e.append((e.id, u, v))
            else:
                intern_e.append((e.id, u, v))
        vlist = sorted(verts)
        non_door = outer_e + intern_e
        if len(non_door) > ENUM_CAP:
            print(f"{mi:>5} {len(vlist):>4} {len(door_e):>4} {len(outer_e):>5} "
                  f"{len(intern_e):>6} {'skip(2^'+str(len(non_door))+')':>8}")
            continue

        fixed_door = [fc_dir[eid] for eid, _u, _v in door_e]  # canonical 방향
        outer_canon = [fc_dir[eid] for eid, _u, _v in outer_e]

        best_do = None
        best_fc = None
        n_do = 0
        n_fc = 0
        n_outer = len(outer_e)
        for bits in product((0, 1), repeat=len(non_door)):
            oriented = list(fixed_door)
            for idx, (_eid, u, v) in enumerate(non_door):
                oriented.append((u, v) if bits[idx] == 0 else (v, u))
            s = apsp_sum(oriented, vlist)
            if s is None:
                continue
            n_do += 1
            if best_do is None or s < best_do:
                best_do = s
            # 외곽이 canonical 과 일치하는가 → S_fc
            outer_ok = all(
                ((outer_e[i][1], outer_e[i][2]) if bits[i] == 0
                 else (outer_e[i][2], outer_e[i][1])) == outer_canon[i]
                for i in range(n_outer)
            )
            if outer_ok:
                n_fc += 1
                if best_fc is None or s < best_fc:
                    best_fc = s

        gain = ""
        if best_do is not None and best_fc is not None and best_do < best_fc:
            gain = "YES"
        print(f"{mi:>5} {len(vlist):>4} {len(door_e):>4} {len(outer_e):>5} "
              f"{len(intern_e):>6} {n_do:>8} {n_fc:>8} "
              f"{('' if best_do is None else f'{best_do:.0f}'):>8} "
              f"{('' if best_fc is None else f'{best_fc:.0f}'):>8} {gain:>5}")


if __name__ == "__main__":
    main()
