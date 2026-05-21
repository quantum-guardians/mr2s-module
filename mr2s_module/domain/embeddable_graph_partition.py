from dataclasses import dataclass

from mr2s_module.domain.graph import Graph
from mr2s_module.domain.score import EmbeddingEstimate


@dataclass
class EmbeddableGraphPartition:
    sub_graphs: list[Graph]
    embedding_estimates: list[EmbeddingEstimate]
    target_k: int | None = None
