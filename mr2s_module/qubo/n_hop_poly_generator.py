from dataclasses import dataclass

from dimod import Vartype
from dimod.higherorder.polynomial import BinaryPolynomial

from mr2s_module.domain import AdjEntry
from mr2s_module.protocols import Graph
from mr2s_module.util import indicator_terms

# 누적 중인 다항식의 항 사전. BinaryPolynomial 은 생성 1회마다 vartype 검증
# (inspect.getfullargspec)과 전 항 재집계(asfrozenset)를 한다. n-hop 재귀는 간선
# 하나를 이을 때마다 다항식을 만들므로 그 비용이 전체를 지배한다. 내부에서는 dict 로
# 누적하고 run() 끝에서 한 번만 BinaryPolynomial 로 옮긴다. 대수는 그대로다.
PolyTerms = dict[frozenset[str], float]

_CONSTANT_TERM: frozenset[str] = frozenset()


def _multiplied(terms: PolyTerms, factor: PolyTerms) -> PolyTerms:
    """terms * factor. factor 는 간선 지시자라 항이 1~2개뿐이라 바깥 루프에 둔다."""
    product: PolyTerms = {}
    for factor_term, factor_coeff in factor.items():
        for term, coeff in terms.items():
            merged = term | factor_term
            product[merged] = product.get(merged, 0.0) + coeff * factor_coeff
    return product


def _add_into(terms: PolyTerms, other: PolyTerms) -> None:
    """other 를 terms 에 제자리 누적. 사본을 만들지 않는다."""
    for term, coeff in other.items():
        terms[term] = terms.get(term, 0.0) + coeff


@dataclass
class NHop:
    n: int
    weight: int


@dataclass
class SmallWorldSpec:
    n_hops: list[NHop]


@dataclass
class NHopPolyGenerator:
    small_world_spec: SmallWorldSpec | None = None

    def _get_n_hop_polynomial(
        self,
        n: int,
        last_vertex: int,
        adj: dict[int, list[AdjEntry]],
        used_vertices: set[int],
        current_polynomial: PolyTerms,
    ) -> PolyTerms:
        if n == 0:
            return current_polynomial

        term_n: PolyTerms = {}

        for entry in adj.get(last_vertex, []):
            if entry.vertex in used_vertices:
                continue

            used_vertices.add(entry.vertex)
            step_poly = (
                {_CONSTANT_TERM: float(entry.weight)}
                if entry.directed
                else indicator_terms(
                    last_vertex, entry.vertex, entry.edge_id, entry.weight
                )
            )
            temp = self._get_n_hop_polynomial(
                n - 1, entry.vertex, adj, used_vertices, step_poly
            )
            _add_into(term_n, temp)
            used_vertices.remove(entry.vertex)

        return _multiplied(term_n, current_polynomial)

    def _get_total_n_hop_polynomial(
        self, n_hop: NHop, vertices: set[int], adj: dict[int, list[AdjEntry]]
    ) -> PolyTerms:
        term_n: PolyTerms = {}
        used_vertices: set[int] = set()
        for vertex in vertices:
            used_vertices.add(vertex)
            temp = self._get_n_hop_polynomial(
                n_hop.n, vertex, adj, used_vertices, {_CONSTANT_TERM: 1.0}
            )
            _add_into(term_n, temp)
            used_vertices.remove(vertex)
        weight = float(n_hop.weight)
        return {term: coeff * weight for term, coeff in term_n.items()}

    def _build_polynomial(
        self, vertices: set[int], adj: dict[int, list[AdjEntry]]
    ) -> PolyTerms:
        terms: PolyTerms = {}
        if self.small_world_spec is None:
            raise ValueError(
                "NHopPolyGenerator requires small_world_spec to build a polynomial"
            )
        for n_hop in self.small_world_spec.n_hops:
            _add_into(terms, self._get_total_n_hop_polynomial(n_hop, vertices, adj))
        # 기존 add_polys 가 매 합산마다 걷어내던 유령 항을 마지막에 한 번 걷어내고
        # scale(-1) 을 겸한다. 중간 항이 0 으로 상쇄돼도 이후 덧셈 결과는 같다.
        return {term: -coeff for term, coeff in terms.items() if abs(coeff) > 1e-12}

    def run(self, graph: Graph) -> BinaryPolynomial:
        if graph.is_empty():
            return BinaryPolynomial({}, Vartype.BINARY)

        vertices = graph.get_vertices()
        adj = graph.get_adjacency_dict()
        return BinaryPolynomial(self._build_polynomial(vertices, adj), Vartype.BINARY)
