from dataclasses import dataclass
from mr2s_module.domain.graph import Graph
from mr2s_module.domain.edge import Edge

@dataclass
class GraphPartitionResult:
    """Face Cycle 이후 결과를 담아주는 명시적 클래스"""
    sub_graphs: list[Graph]
    remaining_edges: list[Edge]

    def outline_of(self, macro_id: int) -> list[Edge]:
        """`macro_id` 거대 면의 외각선(directed boundary) 들을 반환.

        외각선은 sub_graph 가 아니라 `remaining_edges` 에 directed=True 로 보관되므로,
        macro 의 정점 집합(= 그 sub_graph 의 정점들)에 한쪽이라도 닿는
        directed remaining edge 를 모아 외각으로 본다.

        주의: 내부 간선이 0 인 단일 면 macro (singleton) 는 정점 집합이 비어
        결과가 빈 리스트가 된다. 이런 macro 는 sub_graph 만으로는 식별되지 않는다.
        """
        macro_vertices = self.sub_graphs[macro_id].get_vertices()
        return [
            edge for edge in self.remaining_edges
            if edge.directed
            and (edge.id[0] in macro_vertices or edge.id[1] in macro_vertices)
        ]

