import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.config import RunSpec
from experiments.graphs import build_record, graph_path, save_graph
from experiments.run_one import execute, main, write_result
from experiments.solutions import load_solution, reevaluate

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def graph_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("graphs")
    record = build_record(12, 0, 0.3)
    save_graph(graph_path(directory, record.graph_id), record)
    return directory


@pytest.mark.parametrize("use_reduction", [True, False])
def test_execute_records_solution_and_metadata(
    graph_dir: Path, use_reduction: bool
) -> None:
    spec = RunSpec(12, 0, 0.3, "h2", use_reduction, 0)
    result = execute(spec, graph_dir, num_reads=10)

    assert result["status"] == "ok"
    assert result["run_id"] == spec.run_id
    assert result["n_subgraphs"] is not None and result["n_subgraphs"] >= 1
    assert result["qubo_vars_total"] is not None
    assert result["n_edges_contracted"] <= result["n_edges"]
    assert len(result["orientation_bits"]) == result["n_edges"]
    assert len(result["solution"]) == result["n_edges"]
    assert result["strongly_connected"] == (result["apsp_sum"] is not None)

    from experiments.graphs import load_graph

    record = load_graph(graph_path(graph_dir, spec.graph_id))
    restored = reevaluate(load_solution(record, result["orientation_bits"]))
    if result["apsp_sum"] is not None:
        assert restored.apsp_sum == pytest.approx(result["apsp_sum"])
    assert restored.flow_score == pytest.approx(result["flow_score"])


def test_run_is_deterministic_across_processes(graph_dir: Path, tmp_path: Path) -> None:
    # Edge id 는 프로세스 전역 카운터라 같은 프로세스 안에서는 QUBO 변수명이 달라진다.
    # 드라이버는 실행마다 새 프로세스를 띄우므로 그 단위로 결정성을 확인한다.
    run_id = "v12_s0_p30__h2__red__r1"
    bits = []
    for name in ("a", "b"):
        out = tmp_path / name
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.run_one",
                "--run-id",
                run_id,
                "--graph-dir",
                str(graph_dir),
                "--out",
                str(out),
                "--num-reads",
                "10",
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
        )
        bits.append(
            json.loads((out / f"{run_id}.json").read_text())["orientation_bits"]
        )
    assert bits[0] == bits[1]


def test_main_writes_json(graph_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    main(
        [
            "--run-id",
            "v12_s0_p30__h2__nored__r0",
            "--graph-dir",
            str(graph_dir),
            "--out",
            str(out),
            "--num-reads",
            "5",
        ]
    )
    payload = json.loads((out / "v12_s0_p30__h2__nored__r0.json").read_text())
    assert payload["status"] == "ok"
    assert not list(out.glob("*.tmp"))


def test_write_result_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "x.json"
    write_result(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not path.with_suffix(".json.tmp").exists()
