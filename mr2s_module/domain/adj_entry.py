from typing import NamedTuple


class AdjEntry(NamedTuple):
  vertex: int # Could be destination or origin vertex
  weight: float
  directed: bool # if directed is true, then vertex is destination
  edge_id: int # 원본 Edge.id. QUBO 변수명(e_{edge_id}) 생성에 사용.
