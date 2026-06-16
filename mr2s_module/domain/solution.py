from dataclasses import dataclass
from typing import Optional

from dimod import SampleSet

from mr2s_module.domain.edge import Edge
from mr2s_module.domain.graph import Graph
from mr2s_module.domain.score import Score


@dataclass
class Solution:
  # key = Edge.id (인스턴스 고유), value = 방향이 정해진 directed Edge.
  # id 로 키잉하므로 같은 양끝점을 잇는 평행 same-direction 간선도 붕괴하지 않고
  # 따로 보존된다(멀티그래프). 각 directed Edge 가 weight 를 직접 들고 있어
  # flow 계산이 solution.graph 의 id 일치에 의존하지 않는다(축약-복원 시 신규 id 안전).
  edges: dict[int, Edge]
  graph: Graph
  sample_set: SampleSet
  score: Optional[Score] = None
