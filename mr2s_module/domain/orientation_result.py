from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mr2s_module.domain.edge import Edge


class OrientationResult(ABC):
    """간선 방향 전처리 결과의 공통 인터페이스.

    구현체는 자신이 결정한 directed edges 를 `get_edges()` 로 노출한다.
    """

    @abstractmethod
    def get_edges(self) -> list[Edge]: ...


@dataclass
class OrientedEdges(OrientationResult):
    """매크로 분할 없는 단순 간선 방향 결정 결과."""
    edges: list[Edge] = field(default_factory=list)

    def get_edges(self) -> list[Edge]:
        return list(self.edges)
