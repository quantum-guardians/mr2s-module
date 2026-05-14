from dataclasses import dataclass, field
from mr2s_module.domain.graph import Graph
from mr2s_module.domain.edge import Edge


@dataclass
class GraphPartitionResult:
    """Face Cycle 결과.

    - `sub_graphs[i]`     : macro `i` 의 full subgraph (inner undirected + outline directed).
                            공유 boundary 는 양쪽 macro 의 `sub_graphs` 에 같은 인스턴스로 등장 (의도된 중복).
                            inner undirected edge는 중복되지 않아야한다.
    - `remaining_edges`   : 어떤 macro 에도 속하지 않은 간선 — 브리지, 외톨이, 고아 directed.
    """
    sub_graphs: list[Graph] = field(default_factory=list)
    remaining_edges: list[Edge] = field(default_factory=list)

    def outline_of(self, index: int) -> list[Edge]:
        """macro `index` 의 외각선 (directed) 만 필터링해 반환."""
        return [e for e in self.sub_graphs[index].edges if e.directed]

    def directed_edges(self) -> list[Edge]:
        # 공유 boundary 는 양쪽 macro 의 sub_graphs 에 같은 Edge 인스턴스로 등장하므로
        # 인스턴스 dedup 이 필요하다 (Edge 는 identity hash 라 set 으로 충분).
        seen: set[Edge] = set()
        for sg in self.sub_graphs:
            seen.update(e for e in sg.edges if e.directed)
        seen.update(e for e in self.remaining_edges if e.directed)
        return list(seen)

    def get_inner_subgraph(self, index: int) -> Graph:
        """macro `index` 의 내부 간선 (undirected) 만 담은 Graph."""
        return Graph(edges=[e for e in self.sub_graphs[index].edges if not e.directed])

    def get_subgraph(self, index: int) -> Graph:
        """macro `index` 의 full subgraph — `sub_graphs[index]` 와 동일."""
        return self.sub_graphs[index]
