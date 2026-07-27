"""실험이 고정한 door 방향을 보존하는 face-cycle 래퍼 (ISSUE-52 door 조합 스캔).

DnC 는 재귀마다 face_cycle.run(graph) 을 호출하는데 FaceClusterPartition 계열은
무방향 입력만 받는다. 실험에서 door 방향을 미리 박아둔 그래프를 그대로 넘기면
ValueError 로 막히므로, 이 래퍼가

  1) 이미 박힌 방향을 기억하고 임시로 무방향으로 되돌린 뒤 부모 파티션 실행,
  2) 부모 그래프와 결과 subgraph 양쪽에 원래 방향을 다시 박는다.

부모 파티션이 자기 클러스터링 기준으로 새로 방향을 준 간선은 건드리지 않는다.
실험 전용 코드 — 프로덕션(mr2s_module/)에 두지 않는다.
"""

from __future__ import annotations

from mr2s_module.domain import Graph, GraphPartitionResult
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition


class FixedDoorFaceClusterPartition(DoorOnlyFaceClusterPartition):
    """입력에 박힌 방향을 보존하는 door-only 파티션."""

    def run(self, graph: Graph) -> GraphPartitionResult:
        fixed = {
            edge.id: edge.vertices
            for edge in graph.edges.values()
            if edge.directed
        }
        if not fixed:
            return super().run(graph)

        for edge in graph.edges.values():
            if edge.id in fixed:
                edge.directed = False
                edge.vertices = edge.endpoints()
        try:
            result = super().run(graph)
        finally:
            for edge in graph.edges.values():
                if edge.id in fixed:
                    edge.set_direction(*fixed[edge.id])

        for sub_graph in result.sub_graphs:
            for edge in sub_graph.edges.values():
                if edge.id in fixed:
                    edge.set_direction(*fixed[edge.id])
        for edge in result.remaining_edges:
            if edge.id in fixed:
                edge.set_direction(*fixed[edge.id])
        return result
