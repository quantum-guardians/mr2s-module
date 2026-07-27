from dataclasses import dataclass

from dimod import SampleSet

from mr2s_module.domain.graph import Graph
from mr2s_module.domain.score import Score


@dataclass
class Solution:
    edges: dict[int, tuple[int, int]]  # edge id → 방향 (u, v). 평행 간선 독립 보존.
    graph: Graph
    sample_set: SampleSet
    score: Score | None = None
