import math

import pytest

from experiments.graphs import build_record
from experiments.solutions import (
    decode_orientation,
    encode_orientation,
    load_solution,
    reevaluate,
)
from mr2s_module.solver.predefined import create_robbin_solver


def test_bits_roundtrip_and_restore_reproduces_score() -> None:
    record = build_record(30, 5, 0.3)
    solver = create_robbin_solver(use_reduction=False)
    solution = solver.run(load_solution(record, "0" * record.n_edges).graph)
    assert solution.score is not None

    bits = encode_orientation(record, solution.edges.values())
    assert len(bits) == record.n_edges
    assert set(decode_orientation(record, bits)) == set(solution.edges.values())

    restored = load_solution(record, bits)
    score = reevaluate(restored)
    assert math.isclose(score.apsp_sum, solution.score.apsp_sum)
    assert math.isclose(score.flow_score, solution.score.flow_score)
    assert score.strong_connect_rate == 1.0


def test_encode_rejects_incomplete_or_duplicate_orientation() -> None:
    record = build_record(12, 0, 0.0)
    (u, v) = record.edges[0]
    with pytest.raises(ValueError):
        encode_orientation(record, [(u, v)])
    full = [(a, b) for a, b in record.edges]
    with pytest.raises(ValueError):
        encode_orientation(record, [*full, (v, u)])


def test_decode_rejects_bad_bits() -> None:
    record = build_record(12, 0, 0.0)
    with pytest.raises(ValueError):
        decode_orientation(record, "0" * (record.n_edges - 1))
    with pytest.raises(ValueError):
        decode_orientation(record, "2" * record.n_edges)
