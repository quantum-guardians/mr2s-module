from dataclasses import dataclass, field
from typing import Any

from mr2s_module.domain.graph import Graph
from mr2s_module.domain.score import EmbeddingEstimate


@dataclass
class EmbeddableGraphPartition:
    sub_graphs: list[Graph]
    # sub_graphs 와 인덱스 정렬. 임베딩을 추정하지 않는 전략은 None 을 채운다.
    embedding_estimates: list[EmbeddingEstimate | None]
    target_k: int | None = None
    solve_contexts: list[Any] = field(default_factory=list)
