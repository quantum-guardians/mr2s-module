import pytest

from mr2s_module import ApspSumRanker
from mr2s_module.qubo import QAQuboSolver
import mr2s_module.qubo.qa_qubo_solver as qa_module


def test_missing_credentials_raises_runtime_error(monkeypatch):
  def _raise(*args, **kwargs):
    raise ValueError("no credentials")

  monkeypatch.setattr(qa_module, "DWaveSampler", _raise)

  with pytest.raises(RuntimeError, match="credentials"):
    QAQuboSolver(ranker=ApspSumRanker())
