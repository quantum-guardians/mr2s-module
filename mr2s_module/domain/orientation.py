EdgeKey = frozenset


class Orientation:
    """간선 방향 설정: 무방향 간선 키(frozenset) -> (꼬리, 머리)."""

    def __init__(self, directions: dict[EdgeKey, tuple[int, int]] | None = None):
        self._directions: dict[EdgeKey, tuple[int, int]] = dict(directions or {})

    @staticmethod
    def edge_key(u: int, v: int) -> EdgeKey:
        return frozenset({u, v})

    def set_direction(self, tail: int, head: int) -> None:
        self._directions[frozenset({tail, head})] = (tail, head)

    def get_direction(self, edge_key: EdgeKey) -> tuple[int, int]:
        return self._directions[edge_key]

    def has_edge(self, edge_key: EdgeKey) -> bool:
        return edge_key in self._directions

    def flip(self, edge_key: EdgeKey) -> None:
        tail, head = self._directions[edge_key]
        self._directions[edge_key] = (head, tail)

    def edge_keys(self) -> list[EdgeKey]:
        return list(self._directions.keys())

    def directed_edges(self) -> list[tuple[int, int]]:
        return list(self._directions.values())

    def items(self):
        return self._directions.items()

    def copy(self) -> "Orientation":
        return Orientation(self._directions)

    def __len__(self) -> int:
        return len(self._directions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Orientation):
            return NotImplemented
        return self._directions == other._directions
