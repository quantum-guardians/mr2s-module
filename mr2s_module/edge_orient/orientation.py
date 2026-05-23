from typing import Iterator

import networkx as nx


EdgeKey = frozenset


class Orientation:
    """간선 방향 설정: 무방향 간선 키(frozenset) -> (꼬리, 머리)."""

    def __init__(self, directions: dict[EdgeKey, tuple[int, int]] | None = None):
        self._directions: dict[EdgeKey, tuple[int, int]] = dict(directions or {})

    @staticmethod
    def edge_key(u: int, v: int) -> EdgeKey:
        return frozenset({u, v})

    def set_direction(self, tail: int, head: int) -> None:
        self._directions[frozenset({tail, head})] = (tail, head)

    def get_direction(self, edge_key: EdgeKey) -> tuple[int, int]:
        return self._directions[edge_key]

    def has_edge(self, edge_key: EdgeKey) -> bool:
        return edge_key in self._directions

    def flip(self, edge_key: EdgeKey) -> None:
        tail, head = self._directions[edge_key]
        self._directions[edge_key] = (head, tail)

    def edge_keys(self) -> list[EdgeKey]:
        return list(self._directions.keys())

    def directed_edges(self) -> list[tuple[int, int]]:
        return list(self._directions.values())

    def items(self):
        return self._directions.items()

    def copy(self) -> "Orientation":
        return Orientation(self._directions)

    def __len__(self) -> int:
        return len(self._directions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Orientation):
            return NotImplemented
        return self._directions == other._directions


def robbins_orient(base_graph: nx.Graph, start_node: int) -> Orientation:
    """DFS 기반 Robbins 방향 결정.

    트리 간선은 부모→자식, back 간선은 자손→조상으로 향한다.
    `start_node` 와 연결된 컴포넌트의 간선만 처리하며, 비연결 컴포넌트는 호출자가 별도 처리한다.
    명시 스택을 사용하므로 큰 그래프에서도 재귀 한계에 걸리지 않는다.
    """
    orientation = Orientation()
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
            orientation.set_direction(u, v)
            stack.append((v, iter(base_graph.neighbors(v))))
        elif parent[u] != v and order[v] < order[u]:
            orientation.set_direction(u, v)

    return orientation
