from mr2s_module.domain.edge import Edge
from mr2s_module.domain.graph import Graph
from mr2s_module.util import domain_graph_to_networkx

import networkx as nx


class Robbin:
    """단순 dfs 순회하면서 부모관계 정립"""

    def orient(self, graph: Graph) -> list[Edge]:
        if graph.is_empty():
            return []

        nx_graph = domain_graph_to_networkx(graph)

        # 브릿지 존재 시 강한 방향성 불가능 → 방향 결정 포기.
        if nx.has_bridges(nx_graph):
            return []

        adj = graph.get_adjacency_dict()
        visited: set[int] = set()
        parent: dict[int, int] = {}
        order: dict[int, int] = {}
        directed_edges: list[Edge] = []

        # 재귀 대신 명시 스택을 사용. 노드별 인접 반복자를 함께 보관해 자식 처리 후
        # 부모로 돌아왔을 때 다음 인접부터 이어서 본다 (정상적인 DFS 진행 순서 유지).
        start_node = next(iter(graph.get_vertices()))
        order[start_node] = 0
        visited.add(start_node)
        counter = 1
        stack: list[tuple[int, iter]] = [(start_node, iter(adj.get(start_node, [])))]

        while stack:
            u, it = stack[-1]
            entry = next(it, None)
            if entry is None:
                stack.pop()
                continue

            v = entry.vertex
            if v not in visited:
                visited.add(v)
                order[v] = counter
                counter += 1
                parent[v] = u
                directed_edges.append(Edge(u, v, entry.weight, True))
                stack.append((v, iter(adj.get(v, []))))
            elif parent.get(u) != v and order[v] < order[u]:
                directed_edges.append(Edge(u, v, entry.weight, True))

        return directed_edges
