# mr2s-module

`mr2s-module` is a Python library for solving an edge-orientation problem on planar graphs.
Given an undirected graph, the goal is to assign a direction to each edge so that:

- the sum of all-pairs shortest path (APSP) distances is minimized
- the directed graph is strongly connected
- flow is preserved as much as possible at every vertex

The library provides preprocessing, QUBO construction, simulated annealing based search, sample ranking, and final evaluation for that problem.
It includes:

- planar graph preprocessing with `FaceCycle`
- QUBO polynomial generation
- simulated annealing based QUBO solving
- final solution evaluation with multiple metrics

The current pipeline separates:

- sample ranking: `ApspSumRanker`
- final evaluation: `Evaluator`
- returned result container: `Solution`

## Requirements

- Python `>= 3.11`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For test and demo dependencies:

```bash
pip install -e ".[test]"
```

## Core Concepts

### Graph

`Graph` stores the input edge list.

```python
from mr2s_module import Edge, Graph

graph = Graph(edges=[
    Edge(0, 1, 1, False),
    Edge(1, 2, 1, False),
    Edge(2, 0, 1, False),
])
```

### Solution

`Solution` is the main result object produced by the QUBO solver.

- `edges`: selected directed edges
- `graph`: source graph used to build the solution
- `sample_set`: raw annealer samples
- `score`: final evaluation result, attached after evaluation

### Score

`Score` stores the final evaluation metrics.

- `apsp_sum`: APSP-based score on the selected directed graph
- `strong_connect_rate`: fraction of sampled solutions that are strongly connected
- `flow_score`: sum of squared flow imbalance per vertex
- `sample_score`: minimum energy among sampled solutions

## Architecture

The current flow is:

1. build or preprocess a `Graph`
2. generate QUBO terms with one or more polynomial generators
3. solve the QUBO with `SAQuboSolver`
4. rank candidate samples with `ApspSumRanker`
5. evaluate the selected `Solution` with `Evaluator`

Important separation:

- `SolutionRankerProtocol`: for scalar sample selection
- `EvaluatorProtocol`: for final `Solution -> Score`

## Main Components

### Preprocessing

- `FaceCycle`

### QUBO generators

- `FlowPolyGenerator`
- `NHopPolyGenerator`
- `SmallWorldSpec`
- `NHop`

### Solvers

- `SAQuboSolver`
- `QuboMR2SSolver`

### Ranking and evaluation

- `ApspSumRanker`
- `Evaluator`

## Example Usage

```python
from mr2s_module import (
    ApspSumRanker,
    Edge,
    Evaluator,
    FlowPolyGenerator,
    Graph,
    NHop,
    NHopPolyGenerator,
    QuboMR2SSolver,
    SAQuboSolver,
    SmallWorldSpec,
)

graph = Graph(edges=[
    Edge(0, 1, 1, False),
    Edge(1, 2, 1, False),
    Edge(2, 0, 1, False),
])

n_hop_generator = NHopPolyGenerator()
n_hop_generator.small_world_spec = SmallWorldSpec(
    n_hops=[NHop(n=2, weight=1)]
)

solver = QuboMR2SSolver(
    face_cycle=None,
    qubo_solver=SAQuboSolver(ranker=ApspSumRanker()),
    evaluator=Evaluator(),
    poly_generators={FlowPolyGenerator(), n_hop_generator},
)

solution = solver.run(graph)

print(solution.edges)
print(solution.score)
```

## Demo Script

A runnable demo is available at:

- [tests/run_sa_qubo_solver_demo.py](tests/run_sa_qubo_solver_demo.py)

It can:

- generate a planar graph with Delaunay triangulation
- remove a percentage of edges while keeping the graph biconnected
- optionally apply `FaceCycle`
- run `SAQuboSolver`
- print the selected orientation and final score

Example:

```bash
python tests/run_sa_qubo_solver_demo.py \
  --num-points 20 \
  --num-reads 30 \
  --remove-ratio 0.3 \
  --use-face-cycle \
  --target-k 8
```

Available arguments:

- `--num-points`: number of vertices in the generated planar graph
- `--seed`: random seed
- `--weight`: uniform edge weight
- `--remove-ratio`: fraction of edges to remove while preserving biconnectedness
- `--num-reads`: number of SA samples
- `--use-face-cycle`: enable `FaceCycle` preprocessing
- `--target-k`: `FaceCycle` target `k`
