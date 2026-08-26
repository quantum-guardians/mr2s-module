# experiments — 논문 5장 재실험 하네스

n-hop 조합 {2, 3, 4, 2+3, 2+3+4} × 간선 축약 on/off 를 같은 그래프 인스턴스에서 비교하고,
모든 실행의 해(간선 방향)를 저장·복원할 수 있게 한다. 솔버는 DnC + QUBO-SA 한 가지이며
라이브러리(`mr2s_module/`)는 수정하지 않는다.

## 준비

```bash
.venv/bin/python -m pip install -e ".[test,experiments]"
```

## 1. 그래프 생성 (`data/graphs/`, 커밋됨)

```bash
.venv/bin/python -m experiments.graphs            # v∈{100..500} × seed 0–9 × 제거 {0,10,30,50}% = 200개
```

`v{v}_s{seed}_p{pct}.json`: 좌표 `pos`, 간선 `edges`(u<v 정렬), 목표/실제 제거 비율.
제거는 이중연결을 유지하는 탐욕 1패스라 같은 seed 의 p30 제거 집합은 p10 을 포함한다.
`manifest.csv` 가 전 그래프의 이중연결·평면성 검사 결과를 담는다.

## 2. 실행

```bash
# 파일럿 (시간 추정용)
.venv/bin/python -m experiments.run_all --results experiments/results/pilot \
    --vertices 100,200 --seeds 0,1 --reps 1 --workers 10
.venv/bin/python -m experiments.aggregate --results experiments/results/pilot --estimate --workers 12

# 본 실행 (재시작하면 완료된 run_id 는 건너뛴다)
nohup .venv/bin/python -m experiments.run_all --results experiments/results/full --workers 12 \
    > experiments/results/full/driver.log 2>&1 &
```

실행 1회 = 워커 프로세스 1개(`experiments.run_one`). 결과는 `results/<name>/runs/<run_id>.json`
(gitignore) 에 원자적으로 쓰이며 해 전체(`solution`, `orientation_bits`)를 담는다.
`run_id = v100_s0_p30__h2+3__red__r0`. 같은 그래프·같은 반복의 구성들은 같은 `run_seed` 를 쓴다.
타임아웃(`config.TIMEOUT_SEC_BY_VERTICES`)이나 워커 예외는 `status=timeout|error` 로 기록되고
`--retry timeout,error` 로만 재시도한다.

## 3. 집계

```bash
.venv/bin/python -m experiments.aggregate --results experiments/results/full --verify
```

산출물(커밋 대상): `results.csv`, `summary_by_config.csv`, `best_of_by_config.csv`,
`paired_reduction.csv`, `wilcoxon_reduction.csv`, `wilcoxon_hops.csv`, `best_by_graph.csv`,
`best_config_counts.csv`, `solutions/<graph_id>.jsonl`(모든 실행의 방향 비트열), `summary.md`,
`figures/*.pdf`. `--verify` 는 모든 비트열을 복원·재평가해 기록된 점수와 대조한다.

## 해 복원

```python
from experiments.graphs import load_graph
from experiments.solutions import load_solution, reevaluate
record = load_graph(Path("experiments/data/graphs/v100_s0_p30.json"))
solution = load_solution(record, bits)   # bits: solutions/*.jsonl 또는 best_by_graph.csv 의 orientation_bits
print(reevaluate(solution))              # Score(apsp_sum=평균 stretch, ...)
```

## 알아둘 점

- 3-hop 항은 경로를 양방향으로 세면 홀수 최고차 항이 상쇄되어 실제로는 2차 다항식이다.
  그래서 h3·h2+3 은 보조 변수가 없고, 4-hop 만 4차 항이 남아 `make_quadratic` 보조 변수로
  변수 수가 크게 늘어난다.
- 라이브러리 QUBO 솔버 기본값은 2+3 hop 이고 데모 스크립트는 2 hop 단독이다.
- SA 샘플러는 `dwave.samplers.SimulatedAnnealingSampler`(C++, seed 지원), `num_reads=100`.
  BQM 변수 순서를 정렬해 넘기므로 같은 `run_id` 는 프로세스 단위로 같은 해를 낸다.
- `strong_connect_rate` 는 병합·축약 경로에서 {0,1} 만 나오므로 주지표는 `strongly_connected`.
