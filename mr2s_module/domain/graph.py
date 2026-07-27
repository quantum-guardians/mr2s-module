from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from mr2s_module.domain.adj_entry import AdjEntry
from mr2s_module.domain.edge import Edge


@dataclass(init=False)
class Graph:
    edges: dict[int, Edge]

    def __init__(self, edges: dict[int, Edge] | Iterable[Edge] | None = None):
        if edges is None:
            self.edges = {}
        elif isinstance(edges, dict):
            self.edges = cast(dict[int, Edge], edges)
        elif isinstance(edges, Iterable) and not isinstance(edges, str):
            self.edges = {edge.id: edge for edge in edges}
        else:
            raise TypeError(
                f"Graph.edges must be a dict or an iterable of Edge, got {type(edges)!r}"
            )

    def define_edge_direction(self, predefined_edges: Iterable[Edge]):
        """외부 orienter 결과를 id 로 찾아 방향을 in-place 로 박는다."""
        for p_edge in predefined_edges:
            if not p_edge.directed:
                continue
            tail, head = p_edge.vertices
            target = self.edges.get(p_edge.id)
            if target is None:
                raise ValueError(f"Directed edge id {p_edge.id} is not in this graph.")
            target.set_direction(tail, head)

    def is_empty(self):
        return len(self.edges) == 0

    def get_vertices(self) -> set[int]:
        return {v for edge in self.edges.values() for v in edge.vertices}

    def get_adjacency_dict(self) -> dict[int, list[AdjEntry]]:
        adj = defaultdict(list)
        for edge in self.edges.values():
            if edge.directed:
                adj[edge.vertices[0]].append(
                    AdjEntry(edge.vertices[1], edge.weight, True, edge.id)
                )
            else:
                adj[edge.vertices[0]].append(
                    AdjEntry(edge.vertices[1], edge.weight, False, edge.id)
                )
                adj[edge.vertices[1]].append(
                    AdjEntry(edge.vertices[0], edge.weight, False, edge.id)
                )
        return dict(adj)
