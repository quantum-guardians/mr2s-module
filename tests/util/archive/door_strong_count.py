"""공유 간선(door) 강연결 배향 개수 계산 실험.

질문: macro 끼리 공유하는 door 가 n개일 때 2^n 방향 조합 중, macro 안에서
강연결을 보장하는 조합은 몇 개인가? 그 개수를 Tutte T(0,2)(=totally-cyclic
orientation 수) 로 정확히 세고, 개수가 임계 이하면 완전 열거한다.

판정 정의(사용자 확정): door=단방향 arc, 나머지 undirected 간선=양방향 가정
후 mixed-graph 강연결. free 간선으로 연결된 정점을 supernode 로 축약하면
'macro 강연결 ⟺ supernode digraph 강연결' 이므로, valid door 조합 수 =
supernode multigraph 의 T(0,2). (양 끝이 같은 supernode 인 door 는 방향 무관
= 자유변수 → 2배씩)

    PYTHONHASHSEED=0 python tests/util/door_strong_count.py

실험 전용 — 프로덕션(mr2s_module/)에 두지 않는다.
"""

from __future__ import annotations

import random
from collections import deque

import networkx as nx
import numpy as np

from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.robbins_door_bruteforce import (
    SEED,
    build_delaunay,
    prune_keep_biconnected,
)

N_VERTICES = 100
REMOVE_RATIO = 0.4
TARGET_K = 10
ENUM_THRESHOLD = 1 << 22  # 개수 이하면 완전 열거


# ---------------------------------------------------------------------------
# [Union-Find: free 간선으로 supernode 축약]
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# [Tutte T(0,2) = totally-cyclic orientation 수, deletion-contraction]
# ---------------------------------------------------------------------------
def _connected(edges: list[tuple[int, int]], a: int, b: int) -> bool:
    """무방향 multigraph edges 에서 a→b 경로 존재?"""
    adj: dict[int, list[int]] = {}
    for x, y in edges:
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)
    if a not in adj:
        return False
    seen = {a}
    dq = deque([a])
    while dq:
        u = dq.popleft()
        if u == b:
            return True
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                dq.append(w)
    return b in seen


def tutte02(edges: list[tuple[int, int]]) -> int:
    """강연결(totally-cyclic) 배향 수 = T(G;0,2). deletion-contraction.

    - loop → ×2 (y=2), 제거
    - bridge → 0 (x=0)
    - 그 외 e: T(G-e) + T(G/e)
    다리 없는 소형 블록에서만 호출하므로 재귀 폭발 없음(cactus 면 2^k 즉답).
    """
    factor = 1
    es: list[tuple[int, int]] = []
    for a, b in edges:
        if a == b:
            factor *= 2  # loop
        else:
            es.append((a, b))
    if not es:
        return factor
    a, b = es[0]
    rest = es[1:]
    if not _connected(rest, a, b):  # e 가 bridge
        return 0
    deleted = tutte02(rest)
    contracted = [
        (a if x == b else x, a if y == b else y) for (x, y) in rest
    ]  # b→a 병합(선택된 e 한 개 제거)
    return factor * (deleted + tutte02(contracted))


# ---------------------------------------------------------------------------
# [macro → supernode 구조]
# ---------------------------------------------------------------------------
def macro_supernode(sub_graph, canon):
    """macro subgraph → (n_sup, cross_door_edges, intra_doors, cactus, rank).

    cross_door_edges: [(eid, sa, sb)] supernode 사이 door
    intra_doors:      [eid]           같은 supernode 안 door(방향 무관)
    """
    verts = set()
    for e in sub_graph.edges.values():
        u, v = e.endpoints()
        verts.add(u)
        verts.add(v)
    dsu = DSU(verts)
    for e in sub_graph.edges.values():
        if not e.directed:  # free 내부간선으로 축약
            u, v = e.endpoints()
            dsu.union(u, v)
    roots = sorted({dsu.find(v) for v in verts})
    sid = {r: i for i, r in enumerate(roots)}

    cross: list[tuple[int, int, int]] = []
    intra: list[int] = []
    for e in sub_graph.edges.values():
        if not e.directed:
            continue
        t, h = canon[e.id]
        sa, sb = sid[dsu.find(t)], sid[dsu.find(h)]
        if sa == sb:
            intra.append(e.id)
        else:
            cross.append((e.id, sa, sb))

    n_sup = len(roots)
    n_arc = len(cross)
    # cactus 판정: supernode 그래프 biconnected 블록 중 |E|>|V| 존재?
    mg = nx.MultiGraph()
    mg.add_nodes_from(range(n_sup))
    for j, (_eid, sa, sb) in enumerate(cross):
        mg.add_edge(sa, sb, key=j)
    cactus = True
    for block in nx.biconnected_component_edges(mg):
        block = list(block)
        nodes = {x for e in block for x in (e[0], e[1])}
        if len(block) > len(nodes):
            cactus = False
            break
    rank = n_arc - n_sup + 1
    return n_sup, cross, intra, cactus, rank


def supernode_strong(cross: list[tuple[int, int, int]], n_sup: int, mask: int) -> bool:
    """cross door 방향 조합(mask) 에서 supernode digraph 강연결?

    mask 비트 j=0 → arc sa→sb, 1 → sb→sa. n_sup 노드가 서로 도달하면 강연결.
    """
    fwd: dict[int, list[int]] = {i: [] for i in range(n_sup)}
    rev: dict[int, list[int]] = {i: [] for i in range(n_sup)}
    for j, (_eid, sa, sb) in enumerate(cross):
        t, h = (sa, sb) if not (mask >> j) & 1 else (sb, sa)
        fwd[t].append(h)
        rev[h].append(t)

    def reach(adj: dict[int, list[int]]) -> int:
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


def enumerate_valid(
    cross: list[tuple[int, int, int]], n_sup: int
) -> list[int]:
    """cross door 방향 조합 중 강연결 되는 mask 전부(brute)."""
    return [m for m in range(1 << len(cross)) if supernode_strong(cross, n_sup, m)]


def main() -> None:
    raw, _pts = build_delaunay(N_VERTICES, SEED)
    pruned = prune_keep_biconnected(raw, REMOVE_RATIO, SEED)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])
    np.random.seed(SEED)
    random.seed(SEED)
    result = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)

    canon: dict[int, tuple[int, int]] = {}
    for sg in result.sub_graphs:
        for e in sg.edges.values():
            if e.directed:
                canon[e.id] = e.vertices  # (tail, head)

    n_all_doors = len({eid for eid in canon})
    print(
        f"V={pruned.number_of_nodes()} E={pruned.number_of_edges()} "
        f"macros(비어있지않음)={sum(1 for sg in result.sub_graphs if sg.edges)} "
        f"전체 door(고유 eid)={n_all_doors}"
    )
    print()
    print(f"{'macro':>5} {'n_sup':>5} {'cross':>5} {'intra':>5} "
          f"{'rank':>4} {'cactus':>6} {'valid(T02)':>14} {'log2':>7}")

    total_valid_per_macro = []
    used = [mi for mi, sg in enumerate(result.sub_graphs) if sg.edges]
    for new_id, mi in enumerate(used):
        sg = result.sub_graphs[mi]
        n_sup, cross, intra, cactus, rank = macro_supernode(sg, canon)
        cnt = tutte02([(sa, sb) for _e, sa, sb in cross]) * (1 << len(intra))
        total_valid_per_macro.append((new_id, cnt, len(cross) + len(intra)))
        log2 = (cnt.bit_length() - 1) if cnt else float("-inf")
        print(f"{new_id:>5} {n_sup:>5} {len(cross):>5} {len(intra):>5} "
              f"{rank:>4} {str(cactus):>6} {cnt:>14,} {log2:>7}")

    print()
    for mid, cnt, ndoor in total_valid_per_macro:
        space = 1 << ndoor
        frac = cnt / space if space else 0
        print(f"macro {mid}: door {ndoor}개 → 2^{ndoor}={space:,} 중 "
              f"강연결 {cnt:,} ({frac:.3%})")

    # ---- 완전열거(개수 ≤ 임계) : cross door 강연결 방향 패턴 관찰 ----
    print()
    print("=== 완전열거 (강연결 door 방향 패턴) ===")
    for new_id, mi in enumerate(used):
        sg = result.sub_graphs[mi]
        n_sup, cross, intra, cactus, rank = macro_supernode(sg, canon)
        space = 1 << len(cross)
        if space > ENUM_THRESHOLD:
            print(f"macro {new_id}: cross {len(cross)} → 2^{len(cross)} "
                  f"열거범위 초과, 스킵")
            continue
        valids = enumerate_valid(cross, n_sup)
        eids = [e for e, _a, _b in cross]
        print(f"macro {new_id}: cross door eids={eids} "
              f"(intra {len(intra)}개 자유), 강연결 {len(valids)}/{space}")
        for m in valids:
            bits = "".join(str((m >> j) & 1) for j in range(len(cross)))
            print(f"    {bits}")

    # ---- 전역 랜덤 door feasibility MC ----
    # door eid 마다 전역 방향 비트(0=canonical, 1=반전). 한 door 가 두 macro 에
    # 등장해도 canon 이 공통이라 비트가 그대로 양쪽 로컬 비트가 된다.
    macro_cross: list[tuple[list[tuple[int, int, int]], int]] = []
    for mi in used:
        n_sup, cross, _intra, _cactus, _rank = macro_supernode(result.sub_graphs[mi], canon)
        macro_cross.append((cross, n_sup))
    all_door_eids = sorted({e for cross, _ in macro_cross for e, _a, _b in cross})
    print()
    print("=== 전역 랜덤 door feasibility MC ===")
    trials = 2_000_000
    ok = 0
    for _ in range(trials):
        bit = {eid: random.getrandbits(1) for eid in all_door_eids}
        good = True
        for cross, n_sup in macro_cross:
            mask = 0
            for j, (eid, _a, _b) in enumerate(cross):
                if bit[eid]:
                    mask |= 1 << j
            if not supernode_strong(cross, n_sup, mask):
                good = False
                break
        ok += good
    print(f"랜덤 door {len(all_door_eids)}개 → {trials:,} 시행 중 전역 강연결 "
          f"{ok} ({ok / trials:.2e})")


if __name__ == "__main__":
    main()
