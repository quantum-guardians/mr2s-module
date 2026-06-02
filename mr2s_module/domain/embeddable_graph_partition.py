from dataclasses import dataclass, field
from typing import Any

from mr2s_module.domain.graph import Graph
from mr2s_module.domain.score import EmbeddingEstimate


@dataclass
class EmbeddableGraphPartition:
    sub_graphs: list[Graph]
    embedding_estimates: list[EmbeddingEstimate]
    target_k: int | None = None
    solve_contexts: list[Any] = field(default_factory=list)
