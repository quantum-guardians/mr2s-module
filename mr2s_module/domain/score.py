from dataclasses import dataclass


@dataclass
class Score:
  apsp_sum: float
  strong_connect_rate: float
  flow_score: float