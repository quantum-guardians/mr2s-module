from __future__ import annotations

from mr2s_module.edge_orient.iterated_local_search import IteratedLocalSearch
from mr2s_module.evaluator import Evaluator
from mr2s_module.protocols import EvaluatorProtocol
from mr2s_module.solver.base_edge_orientation_solver import BaseEdgeOrientationSolver


class IlsMR2SSolver(BaseEdgeOrientationSolver):
    """Mr2sSolverProtocol을 직접 준수하는 Iterated Local Search 독립형 솔버 wrapper."""

    def __init__(
        self,
        max_iter: int = 30,
        patience: int = 5,
        is_relaxed: bool = False,
        perturb_strength: int = 2,
        evaluator: EvaluatorProtocol = Evaluator(),
    ) -> None:
        ils = IteratedLocalSearch(
            max_iter=max_iter,
            patience=patience,
            is_relaxed=is_relaxed,
            perturb_strength=perturb_strength,
        )
        super().__init__(edge_orienter=ils, evaluator=evaluator)
