# 2026-08-29 — ISSUE-87 Delaunay 외 biconnected planar 계열 생성기와 파일럿

- Date: 2026-08-29
- GitHub Issue: #87
- Status: In progress

## Goal
`experiments.graphs` 에 `--family {delaunay,grid,hexagonal,apollonian}` 을 추가하고, 세 새 계열에
기존 파일럿 구성(v∈{100,200} × seed 0–1 × rep 1 × hop 5종 × 축약 on/off = 160회)을 돌려
Delaunay 파일럿과 비교한다.

## 설계
- graph_id 형식(`v{v}_s{s}_p{p}`)과 config/run_one/aggregate 는 건드리지 않는다. 계열은
  그래프 디렉터리(`data/graphs_{family}`)와 결과 디렉터리(`results/pilot_{family}`)로 구분한다.
- `GraphRecord.family` 필드를 기본값 `"delaunay"` 로 추가한다(기존 JSON 은 기본값으로 로드).
- 생성기 시그니처는 Delaunay 와 같은 `(n, seed) -> (nx.Graph, pos)`. 이중연결 유지 간선 제거는
  `thin_biconnected` 를 그대로 쓴다.
  - grid: rows=round(√n), cols=round(n/rows). seed 는 제거 순서에만 쓰인다(p0 은 seed 무관).
  - hexagonal: `nx.hexagonal_lattice_graph(m, n)` 의 정점 수가 n 에 가장 가까운 (m, n) 을 탐색.
  - apollonian: 삼각형에서 시작해 균등 무작위 면에 정점을 삽입(random Apollonian network).
    좌표는 면 삼각형의 무게중심이라 직선 평면 임베딩이 된다.

## Steps
1. graphs.py 생성기·CLI 추가 → verify: `pytest tests/experiments/test_graphs.py`
2. 계열별 그래프 생성 → verify: manifest 의 biconnected/planar 전부 True
3. 계열별 파일럿 실행·집계 → verify: `summary.md` 생성, ok 비율
4. 비교 요약 작성 → 커밋

## Non-goals
본 실행 규모 재실험, 이중연결이 보장되지 않는 계열(Voronoi/Gabriel/RNG/도로망).
