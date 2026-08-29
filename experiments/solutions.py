"""해(간선 방향) 인코딩·복원·재평가.

방향 비트열은 GraphRecord.edges 순서대로 간선당 1문자다: '0' = (u→v), '1' = (v→u)
(u < v). 그래프 JSON 과 비트열만 있으면 Solution 을 복원해 점수를 다시 계산할 수 있다.
"""

from __future__ import annotations

from collections.abc import Iterable

from experiments.graphs import GraphRecord, to_domain_graph
from mr2s_module.domain import Score, Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.util.sample_set import empty_binary_sample_set


def encode_orientation(
    record: GraphRecord, directed_edges: Iterable[tuple[int, int]]
) -> str:
    direction: dict[tuple[int, int], str] = {}
    for tail, head in directed_edges:
        key = (min(tail, head), max(tail, head))
        if key in direction:
            raise ValueError(f"edge {key} oriented twice")
        direction[key] = "0" if tail < head else "1"
    missing = [edge for edge in record.edges if edge not in direction]
    if missing or len(direction) != len(record.edges):
        raise ValueError(
            f"orientation does not match graph edges "
            f"(missing={len(missing)}, extra={len(direction) - len(record.edges) + len(missing)})"
        )
    return "".join(direction[edge] for edge in record.edges)


def decode_orientation(record: GraphRecord, bits: str) -> list[tuple[int, int]]:
    if len(bits) != len(record.edges):
        raise ValueError(
            f"bit string length {len(bits)} != edge count {len(record.edges)}"
        )
    directed: list[tuple[int, int]] = []
    for (u, v), bit in zip(record.edges, bits, strict=True):
        if bit == "0":
            directed.append((u, v))
        elif bit == "1":
            directed.append((v, u))
        else:
            raise ValueError(f"invalid orientation bit {bit!r}")
    return directed


def load_solution(record: GraphRecord, bits: str) -> Solution:
    """그래프 레코드 + 비트열 → 원본 그래프 기준 Solution (sample_set 은 비어 있음)."""
    graph = to_domain_graph(record)
    directed = decode_orientation(record, bits)
    edges: dict[int, tuple[int, int]] = {}
    for edge, (tail, head) in zip(graph.edges.values(), directed, strict=True):
        if edge.endpoints() != (min(tail, head), max(tail, head)):
            raise RuntimeError("edge order mismatch between record and domain graph")
        edges[edge.id] = (tail, head)
    return Solution(edges=edges, graph=graph, sample_set=empty_binary_sample_set())


def reevaluate(solution: Solution) -> Score:
    return Evaluator().run(solution)
