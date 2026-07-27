from __future__ import annotations

from mr2s_module.edge_orient.robbin import Robbin
from mr2s_module.evaluator import Evaluator
from mr2s_module.protocols import EvaluatorProtocol
from mr2s_module.solver.base_edge_orientation_solver import BaseEdgeOrientationSolver


class RobbinMR2SSolver(BaseEdgeOrientationSolver):
    """Mr2sSolverProtocol을 직접 준수하는 Robbin 독립형 솔버 wrapper."""

    def __init__(self, evaluator: EvaluatorProtocol = Evaluator()) -> None:
        super().__init__(edge_orienter=Robbin(), evaluator=evaluator)
