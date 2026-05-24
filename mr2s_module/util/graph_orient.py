from typing import Iterator

import networkx as nx

from mr2s_module.domain.edge import Edge


def robbins_orient(
    base_graph: nx.Graph, start_node: int
) -> dict[frozenset[int], Edge]:
    """DFS 기반 Robbins 방향 결정.

    트리 간선은 부모→자식, back 간선은 자손→조상으로 향한다.
    `start_node` 와 연결된 컴포넌트의 간선만 처리하며, 비연결 컴포넌트는 호출자가 별도 처리한다.
    명시 스택을 사용하므로 큰 그래프에서도 재귀 한계에 걸리지 않는다.
    가중치는 `base_graph` edge 의 ``weight`` 속성에서 가져오며, 없으면 1.
    """
    edges: dict[frozenset[int], Edge] = {}
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
            weight = base_graph[u][v].get("weight", 1)
            edges[frozenset({u, v})] = Edge(u, v, weight, True)
            stack.append((v, iter(base_graph.neighbors(v))))
        elif parent[u] != v and order[v] < order[u]:
            weight = base_graph[u][v].get("weight", 1)
            edges[frozenset({u, v})] = Edge(u, v, weight, True)

    return edges
