# 2026-09-09 — DnC 임베딩 prefilter degeneracy vs treewidth 실측 검증

- Date: 2026-09-09
- GitHub Issue: #94
- Status: In progress

## Goal

`DegeneracyPruningFaceCyclePartitionStrategy` 의 가짜 임베딩 판정(degeneracy ≤ 타깃 degeneracy 8)을
treewidth 상계 기준으로 바꾸면 실제로 나아지는지 실측으로 답한다. "나아진다"의 정의:

1. **판별력**: 서브그래프 BQM 의 실제 minorminer 임베딩 성공/실패를 더 잘 예측한다(정확도·AUC).
2. **종단 성능**: DnC + QUBO-SA 에서 서브그래프 수·최대 변수·소요 시간·apsp·strong connectivity 가
   좋아지면서, 최종 분할의 서브그래프가 실제로 P16 에 임베딩된다.

## Non-goals

- 라이브러리(`mr2s_module/`) 수정. prefilter 교체·`max_degeneracy` 변경은 결과가 나온 뒤 별도 이슈.
- QPU 실기 검증. 타깃은 이상형 `dwave_networkx.pegasus_graph(16)`.
- 4-hop 계열(보조 변수) 실험.

## Context / Constraints

- 사전 조사(2026-09-09):
  - P16: 노드 5,640 / 커플러 40,484 / 최대 차수 15 / degeneracy 8 / `treewidth_min_degree` 상계 454(20 초).
  - 전체 그래프 BQM(seed 0, p0): degeneracy 는 h2 9~10, h2+3 20~21 로 크기와 무관하게 평평.
    treewidth 상계(min_degree)는 h2 33→84, h2+3 100→259 (v100→v500) 로 크기를 따라감.
    모두 454 미만 → "타깃 treewidth 이하" 조건은 사실상 무조건 통과.
  - #93 실측: h2+3 는 v≤175 임베딩 성공·v≥190 실패, h2 는 v≤400 성공·v≥450 실패.
  - full 실험 baseline(h2+3 v500): 평균 212 서브그래프, 최대 변수 58(축약 44), 242 초, apsp 1.437.
- `treewidth_min_fill_in` 은 881 변수에서 5.8 초라 큰 후보에는 못 쓴다. 기본은 `min_degree`.
- 하네스가 main 에 없어 `feat/ISSUE-85` 위에 스택한다. 워커는 `experiments.run_all` 과 같이 프로세스 단위.
- minorminer 는 `timeout`(초)·`threads`·`random_seed` 를 받는다. 시간 제한 실패 = 임베딩 불가(#93 기준).

## Approach (Checklist)
- [x] **Step 0: Recon** — 전략 코드(`degeneracy_pruning.py`, `embedding_aware.py`), 하네스(`experiments/solvers.py`,
      `run_one.py`), baseline 집계, P16 treewidth 계산 시간 확인.
- [x] **Step 1: 후보 생성 + 지표 + 실측 (`experiments/prefilter_bench.py candidates`)**
  - 그래프: v∈{100,200,300,500} × seed 0 × p∈{0,30}, 축약 전/후(`contract_chains`) → 16 그래프.
  - hop: h2, h2+3. `FaceClusterPartition(target_k=k)` 를 k∈{whole,2,3,4,6,8,12,16,24,32,48,64,96} 로
    돌려 서브그래프를 만들고 k 당 최대 3개(간선 수 최대·중앙·최소)를 뽑는다.
  - 후보마다 BQM 상호작용 그래프의 변수·커플링·최대 차수·degeneracy·treewidth 상계(min_degree; 변수 ≤ 600 이면
    min_fill_in 도)·각 계산 시간을 기록하고, minorminer(P16, timeout 60 초, threads 2, seed 0)로 실측한다.
  - 산출: `experiments/results/prefilter/candidates.csv`.
- [x] **Step 2: 판별력 분석 (`prefilter_bench.py analyze`)**
  - 지표별 AUC(Mann–Whitney), 타깃 유도 임계값(degeneracy 8, treewidth 454)의 정확도·FP·FN,
    단일 임계값 최적 정확도와 그 임계값, seed/그래프 교차 검증(leave-one-graph-out).
  - 산출: `experiments/results/prefilter/metrics.csv`, `summary.md` 초안.
- [ ] **Step 3: 종단 비교 (`prefilter_bench.py e2e`)** — 진행 중
  - 전략 5종: (a) degeneracy ≤ 8, (b) treewidth ≤ 454, (c) treewidth ≤ 100(Step 2 보정), (d) degeneracy ≤ 18(보정),
    (e) couplings ≤ 6,743(보정; 후보 실측에서 판별력이 가장 높아 추가).
    전략은 `DegeneracyPruningFaceCyclePartitionStrategy` 를 상속해 `_estimate_degeneracy` 만 treewidth 로 바꾼
    하네스 내부 서브클래스로 구현하고 `max_degeneracy` 로 임계값을 준다(라이브러리 무수정).
  - 그래프: 임계값을 보정한 seed 0 과 겹치지 않도록 **seed 1** 그래프(v100~v500 × p0/p30)에서 h2/h2+3 × 축약 on/off,
    `RunSpec.run_seed`, num_reads=100. seed 0 의 (a)(b) 사전 실행은 `e2e_seed0_phase1.csv` 로 보관.
  - 기록: n_subgraphs, target_k, qubo_vars_max, elapsed, apsp_sum, strongly_connected, 그리고
    최종 분할 서브그래프 전부의 minorminer 실측(120 초 × 2 회) → `partition_embeddable` 비율.
  - 산출: `experiments/results/prefilter/e2e.csv`, `summary.md` 완성.
- [ ] **Step 4: 정리** — ruff/pyright/pytest 통과, 결과 요약을 이슈 #94 코멘트로 남기고 채택/기각 판단 기록.

## Validation
- **Commands to run:**
  - `.venv/Scripts/python.exe -m experiments.prefilter_bench candidates --workers 8`
  - `.venv/Scripts/python.exe -m experiments.prefilter_bench analyze`
  - `.venv/Scripts/python.exe -m experiments.prefilter_bench e2e --workers 4`
  - `ruff check mr2s_module tests experiments && ruff format --check mr2s_module tests experiments`
  - `pyright experiments/prefilter_bench.py tests/experiments/test_prefilter_bench.py`
  - `.venv/Scripts/python.exe -m pytest -m "not slow" tests/experiments`
- **Expected output:**
  - `candidates.csv` 수백 행, 각 행에 실측 결과(`embeddable`)와 지표값.
  - `metrics.csv` 에 지표별 AUC·정확도. `e2e.csv` 에 전략별 종단 지표와 `partition_embeddable`.
  - 결론 문장: treewidth(타깃 유도/보정) 가 degeneracy 8 대비 판별력과 종단 성능에서 어떤지, 채택 여부.

## Risks & Rollback
- **Risks:**
  - minorminer 시간 제한 실패가 "탐색 실패"일 수 있음(#93 주의). 1 차 60 초 × 2 회 실패 35 건 중 20 건을 300 초로
    재실측했더니 16 건이 성공으로 바뀜 → 실측 기록에 상호작용 그래프를 저장하고 그 인스턴스로 재실측하도록 수정(`8d036e0`).
  - 종단 실행 시간: treewidth 454 arm 은 전체 그래프 SA 로 수렴하므로 v500 h2+3 에서 ~60 초, 문제 없음.
    후보 실측은 최악 60 초 × 후보 수 / 워커 수.
  - 축약 그래프의 평행 간선이 `FaceClusterPartition` 에서 처리되는지 확인 필요(`nx_multigraph` 유틸 존재).
- **Rollback steps:** 실험 스크립트와 결과 파일만 추가하므로 `git revert` 로 충분. 라이브러리 동작 변화 없음.

## Open Questions
- 보정 임계값 T* 를 hop 조합별로 따로 둘지(사전 조사에서 h2/h2+3 경계의 treewidth 가 다를 가능성) — Step 2 결과로 결정.

## Handoff
- `.claude/handoffs/2026-09-09_14-00.md` — 다른 머신에서 이어서 진행하기 위한 상태·명령·남은 일 기록.
