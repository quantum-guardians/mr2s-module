from dataclasses import dataclass


@dataclass
class Score:
  apsp_sum: float
  strong_connect_rate: float
  flow_score: float
  sample_score: float = 0.0

@dataclass
class EmbeddingEstimate:
  num_logical_variables: int
  num_quadratic_couplings: int
  num_physical_qubits: int
  max_chain_length: int
