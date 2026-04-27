from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class AdjEntry(NamedTuple):
  vertex: int # Could be destination or origin vertex
  weight: int
  directed: bool # if directed is true, then vertex is destination