from dataclasses import dataclass, field
from functools import cached_property
from mr2s_module.domain.graph import Graph
from mr2s_module.domain.edge import Edge

@dataclass
class GraphPartitionResult:
    """Face Cycle 이후 결과를 담아주는 명시적 클래스.

    - `sub_graphs[i]`         : macro `i` 의 내부 간선 (undirected) 만 담는 그래프
    - `remaining_edges`       : 외각선(directed) + 브리지/외톨이(undirected) 모두 포함
    - `macro_vertex_sets[i]`  : macro `i` 에 속한 모든 면의 정점 합집합 (외각 전용 정점 포함)
    - `subgraphs[i]`          : (lazy) macro `i` 의 내부 + 외각선 합본 그래프
    """
    sub_graphs: list[Graph]
    remaining_edges: list[Edge]
    macro_vertex_sets: list[set[int]] = field(default_factory=list)

    def outline_of(self, macro_id: int) -> list[Edge]:
        """`macro_id` 거대 면의 외각선(directed boundary) 들을 반환.

        macro 의 *면 정점 집합* `macro_vertex_sets[macro_id]` 와
        directed remaining edge 의 양쪽 끝점을 비교해, 둘 다 포함되면 그 macro 의
        외각으로 본다. 두 macro 가 공유하는 boundary 변은 양쪽에서 모두 반환된다.
        """
        macro_vertices = self.macro_vertex_sets[macro_id]
        return [
            edge for edge in self.remaining_edges
            if edge.directed
            and edge.id[0] in macro_vertices
            and edge.id[1] in macro_vertices
        ]

    def get_inner_subgraph(self, index: int) -> Graph:
        """macro `index` 의 내부 간선만 담은 그래프 (모두 undirected)."""
        return self.sub_graphs[index]

    def get_subgraph(self, index: int) -> Graph:
        """macro `index` 의 내부 간선 + 외각선을 합친 그래프.

        내부 간선은 `directed=False`, 외각선은 `directed=True` 로 보존되어
        하나의 `Graph` 안에 섞여 들어간다.
        """
        return Graph(edges=self.sub_graphs[index].edges + self.outline_of(index))

    @cached_property
    def subgraphs(self) -> list[Graph]:
        """모든 macro 의 (내부 + 외각선) 합본 그래프 리스트 — lazy + cached.

        `partition.subgraphs[i]` 로 macro `i` 의 full view 를 얻을 수 있어
        호출 측에서 일반 인덱싱처럼 쓸 수 있다. 첫 접근 시 한 번 만들고 캐시,
        이후 동일 객체 재사용.
        """
        return [self.get_subgraph(i) for i in range(len(self.sub_graphs))]
