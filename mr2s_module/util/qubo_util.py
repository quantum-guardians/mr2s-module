from dimod import BinaryPolynomial, make_quadratic, BINARY, Vartype
from dwave.samplers import SimulatedAnnealingSampler

sampler = SimulatedAnnealingSampler()

def get_indicator_function(i: int, j: int, weight: int) -> BinaryPolynomial:
  if i == j:
    raise ValueError(f"i and j must be different, but both are {i}")

  if i < j:
    return BinaryPolynomial({(): weight, (f'e_{i}_{j}', ): -weight}, Vartype.BINARY)
  else:
    return BinaryPolynomial({(f'e_{j}_{i}',): weight}, Vartype.BINARY)

def map_binary_poly_to_bqm(polynomial: BinaryPolynomial):
  coeffs = [abs(v) for k, v in polynomial.items() if k != ()]
  max_coeff = max(coeffs) if coeffs else 1.0
  return make_quadratic(polynomial, strength=max_coeff * 2.0, vartype=BINARY)


def build_bqm(polynomial: BinaryPolynomial):
  """다항식 → BQM 변환만 담당하는 분리된 함수."""
  return map_binary_poly_to_bqm(polynomial)

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