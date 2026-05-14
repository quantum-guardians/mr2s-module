# DnC MR2S Solver 성능 병목 분석 보고서

## 분석 대상

- 로그 파일: `docs/logs/DNC_MR2S_SOLVER_PERFORMANCE_RUN.log`
- 로그 라인 수: 3,480
- 입력 그래프: 정점 500개, 간선 1,183개
- 분석 기준: 실제 `DnCMr2sSolver.run()` 실행 구간

## 요약

실제 `DnCMr2sSolver.run()` 실행 시간은 약 22분 5초였다.

가장 큰 병목은 그래프 분할 과정에서 수행되는 embedding estimate다. 전체 실행 시간의 약 93.9%가 partition 단계에서 소비되었고, 그 안에서도 `estimate_required_qubits` 호출이 대부분의 시간을 차지했다.

반대로 face-cycle partition 자체, merge, 방향 적용, scoring, BQM build는 주 병목이 아니다.

## 전체 실행 시간 분해

| 단계 | 시간 | 비중 |
|---|---:|---:|
| Partition phase | 1,244,369.579 ms | 93.90% |
| Subgraph solve phase | 78,233.179 ms | 5.90% |
| Final solve | 2,523.951 ms | 0.19% |
| Score phase | 126.039 ms | 0.01% |
| Merge phase | 1.214 ms | 약 0.00% |
| Apply directions | 0.755 ms | 약 0.00% |
| Total run | 1,325,254.927 ms | 100.00% |

환산하면 다음과 같다.

- 전체 실행: 약 22분 5초
- Partition phase: 약 20분 44초
- Subgraph solve phase: 약 1분 18초

## 가장 큰 병목: 전체 그래프 embedding estimate

실제 `run()` 내부에서 전체 그래프에 대한 embedding estimate가 실패하기까지 약 16분 55초가 걸렸다.

| 그래프 | BQM variables | BQM build | 전체 embedding estimate |
|---|---:|---:|---:|
| 500 vertices / 1,183 edges | 1,183 | 6,281.813 ms | 1,014,764.110 ms |

이 구간 하나만으로 실제 전체 실행 시간의 대부분을 차지한다.

핵심 문제는 전체 그래프가 너무 큰데도 `estimate_required_qubits`를 실제로 호출한다는 점이다. 이 호출은 실패하더라도 매우 오래 걸린다.

따라서 가장 효과적인 개선은 `estimate_required_qubits` 호출 전에 저렴한 prefilter로 너무 큰 그래프를 빠르게 제외하는 것이다.

## Binary Search Partition 분석

실제 `run()` 내부의 target_k search는 총 229,605.241 ms, 약 3분 50초가 걸렸다.

각 attempt에서 face-cycle partition 자체는 비교적 저렴했다. 비용이 커지는 부분은 partition 결과로 나온 subgraph들이 embeddable한지 확인하는 embedding estimate 단계다.

| target_k | Subgraph 수 | Partition 시간 | Embedding estimate 시간 |
|---:|---:|---:|---:|
| 592 | 490 | 1,059.887 ms | 1,201.676 ms |
| 296 | 270 | 881.729 ms | 2,480.573 ms |
| 148 | 153 | 733.462 ms | 3,114.553 ms |
| 74 | 68 | 573.371 ms | 5,671.839 ms |
| 37 | 53 | 534.096 ms | 12,087.351 ms |
| 19 | 37 | 490.992 ms | 17,009.992 ms |
| 10 | 21 | 455.420 ms | 27,768.954 ms |
| 5 | 19 | 399.992 ms | 40,302.435 ms |
| 3 | 20 | 401.787 ms | 36,648.680 ms |
| 2 | 9 | 386.954 ms | 77,399.516 ms |

관찰:

- `target_k`가 작아질수록 subgraph 수는 줄어드는 경향이 있다.
- 대신 각 subgraph가 커지고, 큰 subgraph의 embedding estimate 비용이 급격히 증가한다.
- 이미 feasible한 partition을 찾은 뒤에도 더 작은 `target_k`를 찾기 위해 추가 estimate를 수행한다.
- 런타임을 우선한다면 “가장 작은 target_k”보다 “충분히 좋은 partition”에서 멈추는 전략이 더 적합할 수 있다.

## Embedding Estimate 분포

실제 `run()` 내부의 embedding estimate 호출은 대부분 작고 빠른 subgraph에 대한 호출이지만, 소수의 큰 subgraph가 시간을 지배한다.

실제 `run()` 로그 기준 embedding estimate 통계는 다음과 같다.

| 지표 | 값 |
|---|---:|
| 성공한 embedding estimate 수 | 1,140 |
| 실패한 embedding estimate 수 | 1 |
| 성공 estimate 총합 | 223,656.431 ms |
| 실패 estimate 총합 | 1,014,764.110 ms |
| 성공 estimate 중앙값 | 1.792 ms |
| 성공 estimate p95 | 178.973 ms |
| 성공 estimate p99 | 4,042.901 ms |
| 성공 estimate 최대 | 61,919.606 ms |

대부분의 estimate는 매우 빠르지만, 큰 그래프 몇 개가 전체 시간을 지배한다.

상위 성공 estimate 예시는 다음과 같다.

| Vertices | Edges | Directed edges | BQM variables | 전체 estimate | Estimator only |
|---:|---:|---:|---:|---:|---:|
| 312 | 655 | 121 | 534 | 61,919.606 ms | 60,240.892 ms |
| 151 | 324 | 41 | 283 | 20,591.132 ms | 19,904.470 ms |
| 300 | 626 | 149 | 477 | 18,940.617 ms | 17,487.344 ms |
| 145 | 292 | 80 | 212 | 13,689.253 ms | 13,230.565 ms |
| 118 | 249 | 40 | 209 | 12,089.347 ms | 11,619.927 ms |

## BQM Build 비용

BQM build 비용은 측정 가능하지만 주 병목은 아니다.

실제 `run()` 내부 기준:

| 지표 | 값 |
|---|---:|
| 호출 수 | 1,141 |
| BQM build 총합 | 22,542.460 ms |
| 평균 | 19.757 ms |
| 중앙값 | 1.753 ms |
| p95 | 33.501 ms |
| p99 | 303.568 ms |
| 최대 | 6,281.813 ms |

BQM build는 실제 전체 실행 시간의 약 1.7% 수준이다.

즉 BQM build 최적화보다 `estimate_required_qubits` 호출 횟수와 호출 대상을 줄이는 것이 훨씬 중요하다.

## Subgraph Solve 분석

Subgraph solve는 병렬로 실행되었다.

- Subgraph 수: 9
- Process 수: 8
- 병렬 solve wall time: 77,827.904 ms
- 전체 subgraph solve phase: 78,233.179 ms

개별 subgraph solve 시간:

| Subgraph | 시간 |
|---|---:|
| 312 vertices / 655 edges | 76,575.856 ms |
| 228 vertices / 481 edges | 57,088.256 ms |
| 63 vertices / 138 edges | 18,273.022 ms |
| Small subgraph | 212.256 ms |
| Small subgraph | 81.509 ms |
| Directed-only subgraphs | 약 0.1 ms ~ 5.9 ms |

관찰:

- 병렬 실행이므로 wall time은 가장 오래 걸린 subgraph에 의해 거의 결정된다.
- directed-only subgraph는 QUBO solve를 skip하고 있어 비용이 거의 없다.
- partition 병목을 줄인 뒤에는 가장 큰 subgraph 크기를 줄이는 것이 다음 solve-time 병목 개선 포인트가 될 수 있다.

## 캐싱 효과 평가

실제 `run()` 내부에서 subgraph embedding estimate 캐싱 효과는 크지 않아 보인다.

로그 기반 추정:

- 중복되어 보이는 embedding estimate를 캐시했을 때 절약 가능 시간: 약 4.9초 ~ 6.3초
- 전체 `run()` 대비 비중: 약 0.4% ~ 0.5%

이유:

- 중복 호출은 많지만 대부분 작은 subgraph다.
- 작은 subgraph estimate는 보통 1ms ~ 수십 ms 수준이라 캐싱해도 절대 절약 시간이 작다.

따라서 광범위한 subgraph 캐싱은 복잡도 대비 효과가 작다.

다만 다음 경우에는 제한적 캐싱이 의미 있을 수 있다.

- 큰 graph 또는 큰 subgraph의 실패 결과 캐싱
- 동일한 큰 subgraph가 여러 번 등장하는 워크로드
- 성능 테스트에서 같은 입력을 반복 실행하는 경우

## 개선 우선순위

### 1. `estimate_required_qubits` 호출 전 prefilter 강화

가장 효과가 클 가능성이 높다.

현재 전체 그래프는 BQM variable 1,183개이고, 실패 판정까지 약 16분 55초가 걸렸다. 이런 입력은 실제 embedding estimate를 호출하기 전에 빠르게 제외해야 한다.

prefilter 후보:

- BQM variable 수
- undirected edge 수
- target graph node 수 대비 variable 수
- 과거 로그 기반으로 정한 empirical threshold

### 2. Binary search 종료 조건 완화

현재는 feasible partition을 찾은 뒤에도 더 작은 `target_k`를 찾기 위해 계속 탐색한다.

하지만 작은 `target_k`는 큰 subgraph를 만들고, 큰 subgraph는 embedding estimate 비용을 급격히 키운다.

런타임이 중요하다면 다음 조건 중 하나를 만족할 때 조기 종료할 수 있다.

- 최대 subgraph BQM variable 수가 threshold 이하
- 최대 subgraph edge 수가 threshold 이하
- embedding estimate 총 시간이 일정 예산 이하
- 현재 partition의 subgraph 수와 최대 subgraph 크기가 허용 범위 안에 있음

### 3. 큰 subgraph 크기 균형 개선

Subgraph solve phase는 가장 큰 subgraph에 의해 wall time이 결정된다.

현재 가장 큰 subgraph는 312 vertices / 655 edges이고, solve에 약 76.6초가 걸렸다.

partition 품질을 평가할 때 subgraph 수만 볼 것이 아니라 최대 subgraph 크기와 크기 분산도 함께 봐야 한다.

### 4. 제한적 캐싱

캐싱은 주 병목 해결책은 아니지만 보조 최적화로는 사용할 수 있다.

추천 범위:

- 큰 graph 또는 큰 subgraph의 embedding 실패 결과
- estimate 시간이 일정 threshold를 넘은 항목
- 동일 입력을 반복 실행하는 benchmark 환경

작은 subgraph 전체에 대한 broad cache는 이 로그 기준으로 효과가 작다.
