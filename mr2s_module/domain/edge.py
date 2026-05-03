
class Edge:
  id: tuple[int, int]
  vertices: tuple[int, int] # 방향 결정 시 vertices.first -> vertices.second로 진행. directed가 true 일 때만 사용.
  weight: int
  directed: bool

  def __init__(self, vertex1: int, vertex2: int, weight: int, directed: bool):
    self.id = (min(vertex1, vertex2), max(vertex1, vertex2))
    if directed:
      self.vertices = (vertex1, vertex2)
    else:
      self.vertices = self.id
    self.weight = weight
    self.directed = directed

  def other_vertex(self, vertex: int) -> int:
    if vertex not in self.vertices:
      raise ValueError(f"Vertex {vertex} is not in this edge.")

    v1, v2 = self.vertices
    return v2 if vertex == v1 else v1

  def to_key(self) -> str:
    return f"e_{min(self.vertices)}_{max(self.vertices)}"

