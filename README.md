# mr2s-module

`mr2s-module` is a Python library for solving an edge-orientation problem on planar graphs.
Given an undirected graph, the goal is to assign a direction to each edge so that:

- the sum of all-pairs shortest path (APSP) distances is minimized
- the directed graph is strongly connected
- flow is preserved as much as possible at every vertex

The library provides preprocessing, QUBO construction, simulated annealing based search, sample ranking, and final evaluation for that problem.
It includes:

- planar graph preprocessing with `FaceClusterPartition`
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
3. solve the QUBO with `QuboSolver`
4. rank candidate samples with `ApspSumRanker`
5. evaluate the selected `Solution` with `Evaluator`

Important separation:

- `SolutionRankerProtocol`: for scalar sample selection
- `EvaluatorProtocol`: for final `Solution -> Score`

## Main Components

### Preprocessing

- `FaceClusterPartition`
- `DegreeTwoChainReducer` — collapses degree-2 vertex chains into single edges (see [Chain Reduction](#chain-reduction))

### QUBO generators

- `FlowPolyGenerator`
- `NHopPolyGenerator`
- `SmallWorldSpec`
- `NHop`

### Solvers

- `QuboSolver` (unified QUBO solver; pick a backend via factory methods)
  - `QuboSolver.create_sa_solver(ranker=...)` — simulated annealing backend (local, no credentials)
  - `QuboSolver.create_qa_solver(ranker=...)` — D-Wave quantum annealer backend; requires D-Wave API credentials — set `DWAVE_API_TOKEN` or configure `~/.config/dwave/dwave.conf`
- `QuboMR2SSolver`
- `SAMr2sSolver` (direct simulated annealing on edge orientations)

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
    QuboSolver,
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
    edge_orienter=None,
    qubo_solver=QuboSolver.create_sa_solver(ranker=ApspSumRanker()),
    evaluator=Evaluator(),
    poly_generators={FlowPolyGenerator(), n_hop_generator},
)

solution = solver.run(graph)

print(solution.edges)
print(solution.score)
```

## Chain Reduction

In MR2S edge orientation, a degree-2 vertex has its two incident edges forced into a
consistent direction, so a whole chain of degree-2 vertices is decided by a single bit.
Keeping every chain edge as its own QUBO variable wastes variables, and — more importantly
— it blinds the small-world `NHop` reward: an n-hop path that enters a branch-free
"corridor" dead-ends inside the chain instead of reaching the next hub, so the objective
cannot see hub-to-hub (macro) structure within the same horizon.

`DegreeTwoChainReducer` collapses each degree-2 chain `a - x1 - … - xk - b` into a single
edge `a - b`, then `expand()` restores the directed solution onto the original graph. On a
~300-vertex planar graph this typically halves the QUBO variable count, improves
strong-connectivity success, and keeps APSP quality comparable.

> ⚠️ Weight caveat: a collapsed edge's weight is the **sum** of the original chain weights,
> which distorts the Flow/NHop QUBO and collapses strong connectivity. Solve the reduced
> graph with **unit weight** on collapsed edges; `expand()` re-applies the original weights
> when restoring, so APSP distances are preserved.

### Adapter

`solve_with_chain_reduction` wraps a normal `QuboSolver` call with
reduce → unit-reweight → solve → expand. The returned `Solution.edges` are directed edges
on the **original** graph, so the call site is unchanged:

```python
from mr2s_module import (
    ApspSumRanker, FlowPolyGenerator, Graph, NHop, NHopPolyGenerator,
    QuboSolver, SmallWorldSpec,
)
from mr2s_module.reduction import solve_with_chain_reduction
from mr2s_module.util import add_polys
from mr2s_module.util.qubo_util import map_binary_poly_to_bqm

# Provide your polynomial composition as a callback: (graph) -> QUBO.
def build_qubo(graph: Graph):
    n_hop = NHopPolyGenerator(small_world_spec=SmallWorldSpec(n_hops=[NHop(2, 1)]))
    return map_binary_poly_to_bqm(add_polys(FlowPolyGenerator().run(graph), n_hop.run(graph)))

solver = QuboSolver.create_sa_solver(ranker=ApspSumRanker(), num_reads=80)

# Drop-in replacement for `solver.run(build_qubo(graph), graph)`:
solution = solve_with_chain_reduction(graph, build_qubo, solver)

print(solution.edges)   # directed edges on the original graph
```

If the graph has no degree-2 chains, the adapter solves the original graph directly, so
behavior is identical to not using it. Helper functions `reweight_collapsed_to_unit` and
`expand_solution` are also exported for custom pipelines.

### Benchmark

A runnable experiment that measures the n-hop horizon blind spot (structure metrics M1/M2/M3
plus a real SA comparison) is at [tests/run_chain_reduction_benchmark.py](tests/run_chain_reduction_benchmark.py):

```bash
python tests/run_chain_reduction_benchmark.py --num-reads 60
```

## Demo Script

A runnable demo is available at:

- [tests/run_sa_qubo_solver_demo.py](tests/run_sa_qubo_solver_demo.py)

It can:

- generate a planar graph with Delaunay triangulation
- remove a percentage of edges while keeping the graph biconnected
- optionally apply `FaceClusterPartition`
- run `QuboSolver` (simulated annealing backend)
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
- `--use-face-cycle`: enable `FaceClusterPartition` preprocessing
- `--target-k`: `FaceClusterPartition` target `k`

## Performance Analysis

DnC MR2S solver timing analysis is available at:

- [docs/DNC_MR2S_SOLVER_PERFORMANCE_ANALYSIS.md](docs/DNC_MR2S_SOLVER_PERFORMANCE_ANALYSIS.md)
