from collections.abc import Iterator

import networkx as nx

from mr2s_module.domain.edge import Edge
from mr2s_module.util.nx_multigraph import multi_edge_copies


def robbins_orient(
    base_graph: nx.MultiGraph, start_node: int
) -> dict[int, Edge]:
    """DFS 기반 Robbins 방향 결정.

    트리 간선은 부모→자식, back 간선은 자손→조상으로 향한다.
    평행 간선은 첫 copy 가 DFS 방향을 맡고 나머지 copy 는 weight 를 보고
    누적 flow 불균형이 작아지는 쪽으로 배정해 쌍의 flow 기여를 최소화한다
    (동일 weight 면 완전 상쇄). directed 간선 추가는 강연결을 깨지 않으므로
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

    id 오름차순의 첫 copy 는 DFS 방향(tail→head)을 받는 anchor 로 Robbins
    강연결을 홀로 보장한다. 나머지 copy 는 누적 signed flow
    ``net = Σ(tail→head weight) − Σ(head→tail weight)`` 의 절댓값이 작아지는
    쪽으로 배정해 쌍의 flow 불균형 기여를 최소화한다. weight 가 큰 copy 부터
    처리하는 greedy 라 결정적이며(동일 weight 는 stable sort 로 id 순 유지),
    동일 weight 쌍은 net=0 으로 완전 상쇄된다. weight 가 갈리면 완전 상쇄가
    불가능해 최소 |net| 로 근사한다(2-copy 정확, 다중 copy greedy 근사).
    """
    copies = sorted(multi_edge_copies(base_graph, tail, head).items())
    if not copies:
        return

    anchor_id, anchor_data = copies[0]
    anchor = Edge(tail, head, anchor_data.get("weight", 1), True)
    anchor.id = anchor_id
    edges[anchor_id] = anchor
    net = float(anchor.weight)

    rest = sorted(
        copies[1:],
        key=lambda item: item[1].get("weight", 1),
        reverse=True,
    )
    for edge_id, data in rest:
        weight = data.get("weight", 1)
        if net > 0:
            u, v = head, tail
            net -= weight
        else:
            u, v = tail, head
            net += weight
        oriented = Edge(u, v, weight, True)
        oriented.id = edge_id
        edges[edge_id] = oriented
