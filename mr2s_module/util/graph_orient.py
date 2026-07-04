from collections.abc import Iterator

import networkx as nx

from mr2s_module.domain.edge import Edge


def robbins_orient(
    base_graph: nx.MultiGraph, start_node: int
) -> dict[int, Edge]:
    """DFS 기반 Robbins 방향 결정.

    트리 간선은 부모→자식, back 간선은 자손→조상으로 향한다.
    평행 간선은 첫 copy 가 DFS 방향을 맡고 나머지 copy 는 교대로 반대를 받아
    쌍의 flow 기여가 상쇄된다. directed 간선 추가는 강연결을 깨지 않으므로
    Robbins 보장은 첫 copy 만으로 유지된다.
    `start_node` 와 연결된 컴포넌트의 간선만 처리하며, 비연결 컴포넌트는 호출자가 별도 처리한다.
    명시 스택을 사용하므로 큰 그래프에서도 재귀 한계에 걸리지 않는다.
    가중치는 `base_graph` edge 의 ``weight`` 속성에서 가져오며, 없으면 1.
    반환 dict 와 각 Edge 는 도메인 edge id 를 유지한다.
    """
    edges: dict[int, Edge] = {}
    visited: set[int] = {start_node}
    order: dict[int, int] = {start_node: 0}
    parent: dict[int, int | None] = {start_node: None}
    counter = 1
    stack: list[tuple[int, Iterator[int]]] = [
        (start_node, iter(base_graph.neighbors(start_node)))
    ]

    while stack:
        u, it = stack[-1]
        v = next(it, None)
        if v is None:
            stack.pop()
            continue

        if v not in visited:
            visited.add(v)
            order[v] = counter
            parent[v] = u
            counter += 1
            _orient_parallel_copies(base_graph, u, v, edges)
            stack.append((v, iter(base_graph.neighbors(v))))
        elif parent[u] != v and order[v] < order[u]:
            _orient_parallel_copies(base_graph, u, v, edges)

    return edges


def _orient_parallel_copies(
    base_graph: nx.MultiGraph,
    tail: int,
    head: int,
    edges: dict[int, Edge],
) -> None:
    """(tail, head) 쌍의 모든 평행 copy 에 방향을 배정한다.

    id 오름차순 순회라 결정적이며, 짝수번째 copy 는 DFS 방향,
    홀수번째 copy 는 반대 방향을 받는다.
    """
    for index, (edge_id, data) in enumerate(sorted(base_graph[tail][head].items())):
        u, v = (tail, head) if index % 2 == 0 else (head, tail)
        oriented = Edge(u, v, data.get("weight", 1), True)
        oriented.id = edge_id
        edges[edge_id] = oriented
