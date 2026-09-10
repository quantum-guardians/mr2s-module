# 경계 봉합 파일럿: 현행(A) vs 내부 Y자만 지우기(E)

이슈: [#98](https://github.com/quantum-guardians/mr2s-module/issues/98)
브랜치: `feat/ISSUE-98` — 실험 하네스(`experiments/`)가 아직 main 에 없어서 PR #88(`feat/ISSUE-87`,
그 아래 PR #86) 위에 쌓여 있다. PR 을 열 때 base 를 `feat/ISSUE-87` 로 잡는다.
PR #97(면 분할 속도 개선)을 먼저 합치면 파일럿 시간이 크게 준다 — k-means 초기화가 실행 시간의 80% 다.

## 1. 무엇을 재는가

면 분할(`FaceClusterPartition`)의 경계 봉합(T-join) 규칙만 바꾼 두 변형을 **같은 그래프·같은
seed·같은 hop 구성**으로 짝지어 돌리고, 아래를 비교한다.

| 지표 | 뜻 | 좋은 방향 |
|---|---|---|
| `stretch` | 방향화 전후 거리 비율 평균(APSP, 1.0 이 이론 최적) | 낮을수록 |
| `strongly_connected` | 최종 방향화가 강연결인가 | True |
| `n_subgraphs` | 거대면(부분 QUBO) 수 | — (참고) |
| `qubo_vars_max` | 가장 큰 부분 QUBO 의 변수 수 (임베딩 난이도의 대리) | 낮을수록 |
| `elapsed_sec` | 분할 + k 탐색 + QUBO 풀이 시간 | 낮을수록 |

두 변형:

| | A `legacy` (현행) | E `interior_merge` |
|---|---|---|
| T-join 단말 | 경계 차수가 홀수인 모든 정점 | 내부 홀수 정점만. 외벽 정점은 접지(비용 0 가상 노드) |
| 경로 비용 | 외벽 999,999 / 나머지 1 | 외벽 999,999 / **기존 경계 0** / 나머지 1 |
| 결과 동작 | 홀수 정점 사이에 새 경계를 긋는다 → 평행선 사이가 조각 거대면 | Y자에 매달린 기존 경계를 XOR 로 지운다 → 군집 병합 |
| 경로 간선 합치기 | 합집합 | 대칭차 (T-join 정의) |

E 의 근거는 구조 스윕(Delaunay 160개)에서 조각(≤3면) 거대면이 2.18 → 0.26 개/그래프로 줄고
QUBO 자유 변수가 간선 대비 71.2% → 75.5% 로 늘어난 것이다. **stretch 가 실제로 좋아지는지는
아직 모른다** — 이 파일럿이 그것을 잰다. 거대면이 커지므로(정점 60·k=8: 평균 39 → 53면) k 탐색이
k 를 올려 이득을 상쇄할 가능성도 같이 본다.

코드: `FaceClusterPartition(repair_terminals="interior", boundary_weight=0)`,
하네스 쪽 스위치는 `experiments.solvers.build_solver(..., repair="interior_merge")`.
기본값(`"all"`, `1`)은 기존 `_wall_protected_repair` 를 그대로 타므로 본 실험 재현성은 그대로다.

## 2. 준비

```bash
git fetch origin && git checkout feat/ISSUE-98
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[test,experiments]"
pytest tests/cycle/test_face_cluster_partition_repair.py -q   # 옵션 동작 확인 (수 초)
```

고정 그래프 인스턴스는 `experiments/data/graphs/` 에 이미 있다(v100~v500 × seed 10 × 제거 4단계).

## 3. 실행

### 3-1. 스모크 (약 1분) — 파이프라인이 끝까지 도는지

```bash
python -m experiments.pilot_repair_ae --vertices 100 --seeds 0 --remove 0 --hops h2 --num-reads 20 \
  --out experiments/results/pilot_repair_ae_smoke
```

`legacy`, `interior_merge` 두 줄이 `status=ok` 로 찍히고 마지막에 요약 표가 나오면 된다.

### 3-2. 파일럿 본 실행 (정점 100·200, h2·h3, 제거 0·30%, seed 10개 = 80쌍)

```bash
python -m experiments.pilot_repair_ae --vertices 100 200 --seeds 0-9 --remove 0 30 --hops h2 h3
```

- 결과는 `experiments/results/pilot_repair_ae/runs.jsonl` 에 한 줄씩 붙는다. 중간에 끊겨도 같은
  명령을 다시 치면 끝난 실행은 건너뛴다.
- 소요 시간: 실행당 수십 초 (정점 200 은 1~2분; k 탐색 첫 probe 의 k-means 초기화가 지배적 —
  #96 참고). 80쌍 = 160 실행 ≈ 1~3 시간. 밤에 걸어 두면 된다.
- `--num-reads`(기본 100), `--reps`(기본 1), `--no-reduction`(체인 축약 끄기)로 조절한다.
- 정점 300 이상은 시간이 급증하므로 파일럿 결과를 본 뒤 결정한다.

### 3-3. 요약만 다시 보기

```bash
python -m experiments.pilot_repair_ae --summary
```

`experiments/results/pilot_repair_ae/summary.md` 에도 같은 표가 저장된다.

## 4. 결과 읽는 법

요약 표 한 행 = (정점 수, hop) 그룹. 열 의미:

- **오류 A/E**: 분할 실패 등으로 해가 안 나온 실행 수. E 쪽이 크면 병합으로 커진 거대면이
  k 탐색 상한을 넘긴 것이다 — `qubo_vars_max` 와 같이 본다.
- **강연결 A/E**: 강연결 해를 얻은 수. E 가 낮으면 자유도 증가가 강연결을 해친 것.
- **stretch A / E / Δ**: 둘 다 강연결인 쌍에서만 평균. Δ<0 이면 E 가 개선.
- **E 우세/동률/열세**: 쌍별 stretch 비교. 평균이 아니라 "몇 번 이겼나".
- **p(Wilcoxon)**: 쌍별 부호 순위 검정. scipy 가 있을 때만 계산.
- **부분그래프 / QUBO 최대변수**: E 가 부분 그래프를 덜 만들고 최대 변수는 더 클 것으로 예상.

판단 기준(제안):

| 관찰 | 해석 | 조치 |
|---|---|---|
| Δ<0, p<0.05, 강연결 동등, 오류 동등 | E 가 실제로 낫다 | 정점 300~500 확장 후 기본값 전환 검토 |
| Δ≈0, 오류·강연결 동등 | 자유 변수는 늘었지만 SA 가 못 쓴다 | 옵션으로만 유지, 보고서 향후 연구에 기록 |
| E 오류↑ 또는 강연결↓ | 병합으로 부분 QUBO 가 커져 해가 나빠짐 | `boundary_weight` 를 0 대신 소량(예: 0.5)으로, 또는 `repair_terminals="interior"` 만 단독으로 재시도 |

결과 표는 이슈 #98 에 붙인다. 원 데이터(`runs.jsonl`)는 해 비트열을 포함하므로 재평가가 가능하다.

## 5. 관련 파일

| 파일 | 내용 |
|---|---|
| `mr2s_module/cycle/face_cluster_partition.py` | `repair_terminals`, `boundary_weight` 옵션, `_ground_repair`, `_odd_multiplicity_edges` |
| `experiments/solvers.py` | `REPAIR_OPTIONS`, `build_solver(..., repair=)` |
| `experiments/pilot_repair_ae.py` | 파일럿 러너 + 요약 |
| `tests/cycle/test_face_cluster_partition_repair.py` | 옵션 단위 테스트 (띠·바퀴 고정 fixture) |
