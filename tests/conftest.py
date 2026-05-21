import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)

if root_str in sys.path:
    sys.path.remove(root_str)

sys.path.insert(0, root_str)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--mr2s-solvers",
        action="store",
        default=None,
        help=(
            "Comma-separated solver names for MR2S performance tests, "
            "for example dnc_embedding_aware,dnc_degeneracy_pruning."
        ),
    )
