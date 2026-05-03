from dataclasses import dataclass
from typing import Optional

from dimod import SampleSet

from mr2s_module.domain.graph import Graph
from mr2s_module.domain.score import Score


@dataclass
class Solution:
  edges: set[tuple[int, int]]
  graph: Graph
  sample_set: SampleSet
  score: Optional[Score] = None
