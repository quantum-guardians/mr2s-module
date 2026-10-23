from dimod import BINARY, Vartype
from dimod.higherorder.polynomial import BinaryPolynomial
from dimod.higherorder.utils import make_quadratic
from dwave.samplers import SimulatedAnnealingSampler

sampler = SimulatedAnnealingSampler()


def indicator_terms(
    i: int, j: int, edge_id: int, weight: float
) -> dict[frozenset[str], float]:
    """지시자 다항식의 항 사전. 방향 부호 규약(i<j)의 단일 진실.

    BinaryPolynomial 을 만들지 않고 항만 돌려준다. 재귀 안에서 수십만 번 부르는
    호출부(NHopPolyGenerator)가 객체 생성 비용 없이 쓰기 위한 형태다.
    키를 frozenset 으로 두면 항끼리 합칠 때 집합 합집합 한 번이면 된다.
    """
    if i == j:
        raise ValueError(f"i and j must be different, but both are {i}")

    # 변수명은 edge id 기반(평행 간선마다 독립 변수). i<j 는 부호(방향) 결정용으로만 사용.
    var = frozenset((f"e_{edge_id}",))
    if i < j:
        return {frozenset(): weight, var: -weight}
    return {var: weight}


def get_indicator_function(
    i: int, j: int, edge_id: int, weight: float
) -> BinaryPolynomial:
    return BinaryPolynomial(indicator_terms(i, j, edge_id, weight), Vartype.BINARY)


def map_binary_poly_to_bqm(polynomial: BinaryPolynomial):
    coeffs = [abs(v) for k, v in polynomial.items() if k != ()]
    max_coeff = max(coeffs) if coeffs else 1.0
    return make_quadratic(polynomial, strength=max_coeff * 2.0, vartype=BINARY)


def multiply_polys(
    poly1: BinaryPolynomial, poly2: BinaryPolynomial
) -> BinaryPolynomial:
    new_data = {}
    for term1, coef1 in poly1.items():
        for term2, coef2 in poly2.items():
            # 두 항의 변수들을 합침 (튜플 결합 후 정렬하여 중복 제거)
            new_term = tuple(sorted(set(term1) | set(term2)))
            new_coef = coef1 * coef2

            if new_term in new_data:
                new_data[new_term] += new_coef
            else:
                new_data[new_term] = new_coef

    return BinaryPolynomial(new_data, BINARY)


def add_polys(poly1: BinaryPolynomial, poly2: BinaryPolynomial) -> BinaryPolynomial:
    # 1. 첫 번째 다항식의 항들을 복사 (기본 베이스)
    combined_data = dict(poly1.items())

    # 2. 두 번째 다항식의 항들을 하나씩 꺼내서 더함
    for term, coeff in poly2.items():
        if term in combined_data:
            combined_data[term] += coeff  # 기존 항이 있으면 계수 합산
        else:
            combined_data[term] = coeff  # 없으면 새로 추가

    # 3. (선택 사항) 계수가 0인 항 정리 (유령 항 제거)
    # 너무 작은 값(부동소수점 오차)은 아예 삭제해서 깔끔하게 만듦
    cleaned_data = {t: c for t, c in combined_data.items() if abs(c) > 1e-12}

    return BinaryPolynomial(cleaned_data, poly1.vartype)
