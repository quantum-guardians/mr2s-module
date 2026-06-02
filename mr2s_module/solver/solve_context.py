from dataclasses import dataclass
from typing import Any

import networkx as nx

from mr2s_module.domain import EmbeddingEstimate, Graph


@dataclass
class QuboSolveContext:
  graph: Graph
  bqm: Any
  target_graph: nx.Graph | None = None
  embedding_estimate: EmbeddingEstimate | None = None
