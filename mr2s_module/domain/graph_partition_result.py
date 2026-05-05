from dataclasses import dataclass
from mr2s_module.domain.graph import Graph
from mr2s_module.domain.edge import Edge

@dataclass
class GraphPartitionResult:
    """Face Cycle 이후 결과를 담아주는 명시적 클래스"""
    sub_graphs: list[Graph]
    remaining_edges: list[Edge]

