import json
import sys
from pathlib import Path

import pytest

from experiments.config import RunSpec
from experiments.run_all import DriverOptions, load_done, run_matrix, run_spec

SLEEP_WORKER = "import time, sys; time.sleep(30)"
FAIL_WORKER = "import sys; sys.stderr.write('boom'); sys.exit(3)"


def _opts(tmp_path: Path, **overrides) -> DriverOptions:
    base = {
        "graph_dir": tmp_path / "graphs",
        "results_dir": tmp_path / "results",
        "workers": 2,
        "num_reads": 5,
        "timeout_by_vertices": {100: 1},
    }
    base.update(overrides)
    return DriverOptions(**base)


def _fake_worker(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    monkeypatch.setattr(
        "experiments.run_all.worker_command",
        lambda spec, opts: [sys.executable, "-c", code],
    )


def _ok_worker_code(opts: DriverOptions) -> str:
    return (
        "import json, sys, pathlib; run_id = sys.argv[1]; "
        f"out = pathlib.Path({str(opts.runs_dir)!r}); out.mkdir(parents=True, exist_ok=True); "
        "(out / (run_id + '.json')).write_text(json.dumps({'run_id': run_id, 'status': 'ok'}))"
    )


def test_load_done_respects_retry_statuses(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "a.json").write_text(json.dumps({"status": "ok"}))
    (runs / "b.json").write_text(json.dumps({"status": "timeout"}))
    (runs / "c.json").write_text("not json")
    assert load_done(runs) == {"a", "b", "c"}
    assert load_done(runs, {"timeout"}) == {"a"}


def test_timeout_writes_timeout_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = _opts(tmp_path)
    _fake_worker(monkeypatch, SLEEP_WORKER)
    spec = RunSpec(100, 0, 0.0, "h2", True, 0)
    assert run_spec(spec, opts) == "timeout"
    record = json.loads((opts.runs_dir / f"{spec.run_id}.json").read_text())
    assert record["status"] == "timeout"
    assert record["timeout_sec"] == 1


def test_worker_crash_writes_error_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = _opts(tmp_path)
    _fake_worker(monkeypatch, FAIL_WORKER)
    spec = RunSpec(100, 0, 0.0, "h2", False, 0)
    assert run_spec(spec, opts) == "error"
    record = json.loads((opts.runs_dir / f"{spec.run_id}.json").read_text())
    assert record["status"] == "error"
    assert "boom" in record["error_message"]


def test_run_matrix_skips_done_and_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = _opts(tmp_path)
    specs = [RunSpec(100, 0, 0.0, "h2", r, 0) for r in (True, False)]
    opts.runs_dir.mkdir(parents=True)
    (opts.runs_dir / f"{specs[0].run_id}.json").write_text(json.dumps({"status": "ok"}))

    dry = run_matrix(specs, DriverOptions(**{**opts.__dict__, "dry_run": True}))
    assert dry == {"skipped": 1}

    code = _ok_worker_code(opts)
    monkeypatch.setattr(
        "experiments.run_all.worker_command",
        lambda spec, o: [sys.executable, "-c", code, spec.run_id],
    )
    counts = run_matrix(specs, opts)
    assert counts == {"skipped": 1, "ok": 1}
    assert run_matrix(specs, opts) == {"skipped": 2}
