class Edge:
  id: frozenset[int]
  vertices: tuple[int, int]  # directed=True 면 (tail, head), 아니면 sorted endpoints.
  weight: int
  directed: bool

  def __init__(self, vertex1: int, vertex2: int, weight: int, directed: bool):
    self.id = frozenset({vertex1, vertex2})
    self._endpoints = (vertex1, vertex2) if vertex1 <= vertex2 else (vertex2, vertex1)
    if directed:
      self.vertices = (vertex1, vertex2)
    else:
      self.vertices = self._endpoints
    self.weight = weight
    self.directed = directed

  def endpoints(self) -> tuple[int, int]:
    """무방향 정렬된 (u, v). self-loop 면 (v, v)."""
    return self._endpoints

  def other_vertex(self, vertex: int) -> int:
    if vertex not in self.vertices:
      raise ValueError(f"Vertex {vertex} is not in this edge.")

    v1, v2 = self.vertices
    return v2 if vertex == v1 else v1

  def to_key(self) -> str:
    u, v = self._endpoints
    return f"e_{u}_{v}"

  def flip(self) -> "Edge":
    """방향이 뒤집힌 새 Edge 반환. ILS hypothesis 변형에서 사용."""
    return Edge(self.vertices[1], self.vertices[0], self.weight, self.directed)
