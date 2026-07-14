"""Door-only 경계 배향 실험용 partition (ISSUE-52 외곽선 힌트 검증).

FaceClusterPartition 의 간이 카피: 서로 다른 macro 사이 공유 경계 간선(door)만
directed 로 pre-orient 하고, 외곽(outer) 간선과 macro 내부 boundary 간선은
undirected free QUBO 변수로 남긴다. 클러스터링·T-join 수리·ghost 필터·2색칠은
부모 구현을 그대로 재사용한다 — door 의 방향 자체는 여전히 2색칠이 결정해야
하므로 제거할 수 없다.

door 판별: `_ComponentPartition.macro_outline_keys` 에서 서로 다른 macro 2곳에
등장하는 edge id. outer 간선(면 1개)과 intra-macro repair 간선(양쪽 면이 같은
macro)은 owning macro 가 1곳뿐이라 구분된다. directed_steps 를 door 로만
필터하면 부모 `run()` 의 outline 분기가 나머지를 undirected 로 방출하고,
undirected overlap validator 는 owning macro 1곳 조건 덕에 그대로 통과한다.

실험 전용 코드 — 프로덕션(mr2s_module/cycle/)에 두지 않는다.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from mr2s_module.cycle.face_cluster_partition import (
    FaceClusterPartition,
    _ComponentPartition,
)


class DoorOnlyFaceClusterPartition(FaceClusterPartition):
    """door(인접 macro 공유 경계)만 directed, 나머지 boundary 는 free."""

    def _partition_component(self, component: nx.Graph) -> _ComponentPartition:
        partition = super()._partition_component(component)

        owner_count: Counter[int] = Counter()
        for outline_keys in partition.macro_outline_keys:
            for edge_id in outline_keys:
                owner_count[edge_id] += 1
        door_ids = {edge_id for edge_id, count in owner_count.items() if count >= 2}

        return _ComponentPartition(
            macro_internal_edges=partition.macro_internal_edges,
            macro_outline_keys=partition.macro_outline_keys,
            directed_steps={
                step for step in partition.directed_steps if step[0] in door_ids
            },
        )
