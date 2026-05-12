from mr2s_module import (
  ApspSumRanker,
  BalancedFaceGraphClusterer,
  Edge,
  EmbeddingEstimate,
  Evaluator,
  FlowPolyGenerator,
  Graph,
  KMeansFaceClusterer,
  NHop,
  NHopPolyGenerator,
  QuboMR2SSolver,
  SAMR2SSolver,
  SAQuboSolver,
  SmallWorldSpec,
  SnowballFaceClusterer,
)


def test_top_level_exports_support_component_composition() -> None:
    assert Graph is not None
    assert BalancedFaceGraphClusterer is not None
    assert Edge is not None
    assert EmbeddingEstimate is not None
    assert QuboMR2SSolver is not None
    assert ApspSumRanker is not None
    assert Evaluator is not None
    assert FlowPolyGenerator is not None
    assert KMeansFaceClusterer is not None
    assert NHopPolyGenerator is not None
    assert NHop is not None
    assert SmallWorldSpec is not None
    assert SAQuboSolver is not None
    assert SAMR2SSolver is not None
    assert SnowballFaceClusterer is not None
