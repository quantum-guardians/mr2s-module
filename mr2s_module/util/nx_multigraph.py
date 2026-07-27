from collections.abc import Mapping
from typing import cast

import networkx as nx

EdgeAttrs = Mapping[str, float]
EdgeCopies = Mapping[int, EdgeAttrs]


def multi_edge_copies(graph: nx.MultiGraph, u: int, v: int) -> EdgeCopies:
    """(u, v) 의 평행 copy 들을 edge id → 속성 매핑으로 돌려준다.

    networkx 스텁은 `graph[u][v]` 를 단순 그래프의 속성 dict(`dict[str, Any]`) 로 좁히지만
    MultiGraph 에서는 key → 속성 dict 다. 이 프로젝트의 MultiGraph 는 key 로 도메인 edge id
    를 쓰므로 그 계약을 여기서 한 번만 선언한다 (런타임 동작 없음).
    """
    return cast(EdgeCopies, graph[u][v])
