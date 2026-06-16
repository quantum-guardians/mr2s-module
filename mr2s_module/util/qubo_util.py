from dimod import BinaryPolynomial, make_quadratic, BINARY, Vartype
from dwave.samplers import SimulatedAnnealingSampler

sampler = SimulatedAnnealingSampler()

def get_indicator_function(i: int, j: int, weight: int, var_key: str) -> BinaryPolynomial:
  """정점 i 에서의 outflow indicator. var_key 는 간선의 QUBO 변수명(Edge.to_key()).

  변수 x(var_key)=1 은 간선이 max→min 방향임을 뜻한다(process_solution 과 동일 규약).
  indicator 는 간선이 i 에서 나갈 때 weight, 들어올 때 0. i<j 인지로 부호 규약만 정하고
  변수 이름 자체는 var_key 를 써서 같은 양끝점의 평행간선을 서로 다른 변수로 분리한다.
  """
  if i == j:
    raise ValueError(f"i and j must be different, but both are {i}")

  if i < j:
    return BinaryPolynomial({(): weight, (var_key,): -weight}, Vartype.BINARY)
  else:
    return BinaryPolynomial({(var_key,): weight}, Vartype.BINARY)

def map_binary_poly_to_bqm(polynomial: BinaryPolynomial):
  coeffs = [abs(v) for k, v in polynomial.items() if k != ()]
  max_coeff = max(coeffs) if coeffs else 1.0
  return make_quadratic(polynomial, strength=max_coeff * 2.0, vartype=BINARY)

def multiply_polys(
    poly1: BinaryPolynomial,
    poly2: BinaryPolynomial
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
      combined_data[term] += coeff # 기존 항이 있으면 계수 합산
    else:
      combined_data[term] = coeff  # 없으면 새로 추가

  # 3. (선택 사항) 계수가 0인 항 정리 (유령 항 제거)
  # 너무 작은 값(부동소수점 오차)은 아예 삭제해서 깔끔하게 만듦
  cleaned_data = {t: c for t, c in combined_data.items() if abs(c) > 1e-12}

  return BinaryPolynomial(cleaned_data, poly1.vartype)