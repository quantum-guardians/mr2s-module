"""Macro 클러스터링 적용 후 door 배향 성공 검증 (robbins_door_bruteforce 후속).

앞선 실험(robbins_door_bruteforce.py)은 면 단위로 door 를 배향해 면-dual 홀수
사이클 때문에 유효 조합 0 이었다. 여기서는 프로덕션 FaceClusterPartition 을
그대로 적용한다: 면들을 macro 로 클러스터링 → macro 간 경계(door)만 방향 부여
(merged_dual 2색칠) → macro 내부/외곽 간선은 free.

핵심 질문: macro 로 묶으면 각 macro 가 door 고정 방향 하에서 실제로 강연결
배향 가능한가? 각 sub_graph(=macro)에 대해 directed door = arc 1개, undirected
free = arc 2개로 mixed→digraph 변환 후 two-pass BFS 로 단일 SCC 판정한다.

면 단위(대조군)와 달리 macro 병합은 홀수 면-사이클을 macro 내부로 흡수해
merged_dual 을 bipartite 로 만들 수 있다 — 그 효과를 실측한다.

실험 전용. 직접 실행: `python tests/util/robbins_macro_cluster.py`
"""

from __future__ import annotations

from collections import deque

import numpy as np

from mr2s_module.cycle import FaceClusterPartition
from mr2s_module.domain import Edge, Graph
from tests.util.robbins_door_bruteforce import (
    SEED,
    build_delaunay,
    prune_keep_biconnected,
)


def nx_to_domain(nx_graph) -> Graph:
    """무방향 nx 그래프 → domain Graph (weight=1, undirected)."""
    return Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in nx_graph.edges()])


def _reachable(adj: dict[int, list[int]], start: int) -> set[int]:
    seen = {start}
    dq = deque([start])
    while dq:
        x = dq.popleft()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                dq.append(y)
    return seen


def macro_strongly_orientable(macro: Graph) -> bool:
    """macro(sub_graph)가 현재 방향 고정 하에서 강연결 완성 가능한지 판정.

    directed door → arc 1개, undirected free → 양방향 arc 2개. two-pass BFS.
    """
    fwd: dict[int, list[int]] = {}
    bwd: dict[int, list[int]] = {}
    verts: set[int] = set()

    def add_arc(a: int, b: int) -> None:
        fwd.setdefault(a, []).append(b)
        bwd.setdefault(b, []).append(a)

    for edge in macro.edges.values():
        u, v = edge.endpoints()
        verts.add(u)
        verts.add(v)
        if edge.directed:
            a, b = edge.vertices  # (tail, head)
            add_arc(a, b)
        else:
            add_arc(u, v)
            add_arc(v, u)

    if not verts:
        return True
    start = next(iter(verts))
    if _reachable(fwd, start) != verts:
        return False
    return _reachable(bwd, start) == verts


def main() -> None:
    # 앞 실험과 동일한 V=200, remove 0.4 인스턴스.
    raw, _pts = build_delaunay(200, SEED)
    pruned = prune_keep_biconnected(raw, 0.4, SEED)
    graph = nx_to_domain(pruned)

    # 결정적 클러스터링 (프로덕션은 np.random 사용).
    np.random.seed(SEED)
    result = FaceClusterPartition(target_k=10).run(graph)

    macros = result.sub_graphs
    directed = result.directed_edges()

    print("=" * 60)
    print(f"입력 간선 수     : {len(graph.edges)}")
    print(f"macro 수         : {len(macros)}")
    print(f"directed door 수 : {len(directed)} (고유 edge id)")
    print(f"remaining(외곽 등): {len(result.remaining_edges)}")
    print("=" * 60)

    ok = 0
    fail_sizes: list[int] = []
    for m in macros:
        if macro_strongly_orientable(m):
            ok += 1
        else:
            fail_sizes.append(len(m.edges))

    print(f"강연결 완성 macro: {ok} / {len(macros)}")
    if fail_sizes:
        print(f"실패 macro 간선수: {sorted(fail_sizes)}")
    else:
        print("→ 모든 macro 가 door 방향 하에서 강연결 완성 가능 (door 배향 성공)")

    # 대조 요약: 면 단위였으면 유효 0 이었음(면-dual non-bipartite).
    empty = len(directed) == 0
    print(
        "merged_dual bipartite 여부(추정): "
        + ("FALSE — 방향 미부여(빈 partition)" if empty else "TRUE — door 방향 부여됨")
    )


if __name__ == "__main__":
    main()
