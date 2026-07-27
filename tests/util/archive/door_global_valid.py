"""전역에서 valid 한 door 배향이 몇 개나 존재하는가?

per-macro 로는 valid door 배향이 많지만(part1), door 는 두 macro 가 공유하므로
전역 일관성 제약이 걸린다. '모든 macro 동시 강연결' 을 만족하는 door 배향을
백트래킹으로 전수(또는 상한까지) 세어, 2색칠 미러쌍(2개) 말고 **진짜 자유가
있는지** 판정한다.

    - 전역 valid == 2  → door 는 사실상 강제(미러쌍), 품질 레버 없음
    - 전역 valid  > 2  → 자유 존재 → valid 중 APSP 최소 고를 여지

    PYTHONHASHSEED=0 python tests/util/door_global_valid.py [V] [K]

실험 전용.
"""

from __future__ import annotations

import random
import sys
from collections import deque

import numpy as np

from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.door_strong_count import macro_supernode
from tests.util.robbins_door_bruteforce import (
    SEED,
    build_delaunay,
    prune_keep_biconnected,
)

N_VERTICES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REMOVE_RATIO = 0.4
TARGET_K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
COUNT_CAP = 1_000_000  # 이 이상이면 '자유 큼' 으로 조기 종료


def macro_strong(cross, n_sup, bit) -> bool:
    """door 전역 비트 dict(bit[eid]: 0=canonical,1=반전) 에서 이 macro 강연결?"""
    fwd = {i: [] for i in range(n_sup)}
    rev = {i: [] for i in range(n_sup)}
    for eid, sa, sb in cross:
        t, h = (sa, sb) if not bit[eid] else (sb, sa)
        fwd[t].append(h)
        rev[h].append(t)

    def reach(adj):
        seen = {0}
        dq = deque([0])
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    dq.append(w)
        return len(seen)

    return reach(fwd) == n_sup and reach(rev) == n_sup


def main() -> None:
    raw, _pts = build_delaunay(N_VERTICES, SEED)
    pruned = prune_keep_biconnected(raw, REMOVE_RATIO, SEED)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])
    np.random.seed(SEED)
    random.seed(SEED)
    result = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)

    canon = {
        e.id: e.vertices
        for sg in result.sub_graphs for e in sg.edges.values() if e.directed
    }
    used = [mi for mi, sg in enumerate(result.sub_graphs) if sg.edges]
    macros = []  # (cross, n_sup, doorset)
    for mi in used:
        n_sup, cross, _intra, _cactus, _rank = macro_supernode(result.sub_graphs[mi], canon)
        if cross:  # cross door 없으면 제약 없음 → 제외
            macros.append((cross, n_sup, {e for e, _a, _b in cross}))

    all_doors = sorted({e for _c, _n, ds in macros for e in ds})
    # door 순서: 각 macro 가 최대한 빨리 '완성' 되도록, macro 별 door 를 이어 붙임
    order: list[int] = []
    seen: set[int] = set()
    for _c, _n, ds in sorted(macros, key=lambda m: len(m[2])):
        for e in sorted(ds):
            if e not in seen:
                seen.add(e)
                order.append(e)
    pos = {e: i for i, e in enumerate(order)}

    # 각 macro 가 '몇 번째 door 까지 배정되면 완성' 되는지 = 그 macro door 의 최대 pos
    triggers: dict[int, list[int]] = {i: [] for i in range(len(order))}
    for mid, (_c, _n, ds) in enumerate(macros):
        last = max(pos[e] for e in ds)
        triggers[last].append(mid)

    print(f"V={pruned.number_of_nodes()} door={len(all_doors)} macros={len(macros)}")

    bit: dict[int, int] = {}
    count = 0
    capped = False

    def dfs(i: int) -> None:
        nonlocal count, capped
        if capped:
            return
        if i == len(order):
            count += 1
            if count >= COUNT_CAP:
                capped = True
            return
        eid = order[i]
        for b in (0, 1):
            bit[eid] = b
            ok = True
            for mid in triggers[i]:  # 이 배정으로 완성된 macro 검사
                cross, n_sup, _ds = macros[mid]
                if not macro_strong(cross, n_sup, bit):
                    ok = False
                    break
            if ok:
                dfs(i + 1)
            if capped:
                return
        bit.pop(eid, None)

    sys.setrecursionlimit(10000)
    dfs(0)
    tag = f"≥{COUNT_CAP:,} (자유 큼)" if capped else f"{count:,}"
    print(f"전역 valid door 배향 수 = {tag}")
    if not capped:
        print("  2 이면 미러쌍뿐(강제), >2 이면 자유 존재")
        print(f"  log2 ≈ {(count.bit_length() - 1) if count else 0}")


if __name__ == "__main__":
    main()
