# pandas 3.0 은 타입 정보가 부분적이라 DataFrame/Series 연산이 pyright 에서 대량 오탐을 낸다.
# 이 파일은 런타임 테스트(tests/experiments/test_aggregate.py)로 검증한다.
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportCallIssue=false, reportReturnType=false
"""실행 결과 집계: runs/*.json → results.csv, 요약표, 짝지은 비교, 검정, 최선 해, 그림.

산출물 (results/<name>/):
  results.csv              실행 1행 (해 제외)
  summary_by_config.csv    (v, 제거비율, hop, 축약) 별 성공률·stretch 평균±std·best-of·시간
  paired_reduction.csv     같은 (graph, hop, rep) 의 축약 on/off 차이
  wilcoxon_reduction.csv   (v, hop) 별 축약 on/off Wilcoxon 부호순위 검정
  wilcoxon_hops.csv        (v, 축약) 별 h2 대비 다른 hop 의 Wilcoxon 검정
  best_by_graph.csv        그래프별 전체 실행 중 최선 해 (비트열 포함)
  best_config_counts.csv   최선 해를 낸 구성의 빈도
  solutions/<graph>.jsonl  그래프별 모든 실행의 방향 비트열 (복원용)
  summary.md               위 표들의 마크다운 판
  figures/*.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments import config
from experiments.config import RunSpec, iter_run_specs
from experiments.graphs import DEFAULT_GRAPH_DIR, graph_path, load_graph
from experiments.solutions import load_solution, reevaluate

GROUP = ["vertices", "remove_ratio_target", "hop_key", "use_reduction"]
HOP_ORDER = list(config.HOP_SETS)
# dataviz 기본 팔레트의 categorical slot 1..5 (고정 순서, hop 마다 항상 같은 색).
HOP_COLORS = dict(
    zip(HOP_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"], strict=True)
)
REDUCTION_STYLE = {True: "-", False: "--"}
REDUCTION_LABEL = {True: "축약 on", False: "축약 off"}
DROP_FROM_CSV = ("solution", "orientation_bits", "subgraph_sizes")


# --- 수집 -------------------------------------------------------------------


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        payload.pop("solution", None)
        sizes = payload.get("subgraph_sizes")
        payload["subgraph_sizes"] = json.dumps(sizes) if sizes is not None else None
        rows.append(payload)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for column in ("apsp_sum", "sample_score", "elapsed_sec", "flow_score"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "strongly_connected" not in df:
        df["strongly_connected"] = False
    df["strongly_connected"] = df["strongly_connected"].fillna(False).astype(bool)
    df["ok"] = df["status"] == "ok"
    df["hop_key"] = pd.Categorical(df["hop_key"], categories=HOP_ORDER, ordered=True)
    return df.sort_values(
        [
            "rep",
            "vertices",
            "graph_seed",
            "remove_ratio_target",
            "hop_key",
            "use_reduction",
        ]
    ).reset_index(drop=True)


def write_results_csv(df: pd.DataFrame, path: Path) -> None:
    df.drop(columns=[c for c in DROP_FROM_CSV if c in df]).to_csv(path, index=False)


# --- 요약 -------------------------------------------------------------------


def _sc(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["ok"] & df["strongly_connected"]]


def summarize_by_config(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(GROUP, observed=True)
    summary = grouped.agg(
        n_runs=("run_id", "size"),
        n_ok=("ok", "sum"),
        n_timeout=("status", lambda s: int((s == "timeout").sum())),
        n_error=("status", lambda s: int((s == "error").sum())),
        n_sc=("strongly_connected", "sum"),
        time_mean=("elapsed_sec", "mean"),
        time_std=("elapsed_sec", "std"),
        n_subgraphs_mean=("n_subgraphs", "mean"),
        qubo_vars_mean=("qubo_vars_total", "mean"),
        n_edges_mean=("n_edges", "mean"),
        n_edges_solved_mean=("n_edges_solved", "mean"),
    ).reset_index()
    summary["sc_rate"] = summary["n_sc"] / summary["n_runs"]
    sc_stats = (
        _sc(df)
        .groupby(GROUP, observed=True)["apsp_sum"]
        .agg(apsp_mean="mean", apsp_std="std", apsp_min="min", apsp_max="max")
        .reset_index()
    )
    return summary.merge(sc_stats, on=GROUP, how="left")


def best_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """(graph, hop, 축약) 별 반복 중 최선(강연결 해 중 stretch 최소)."""
    sc = _sc(df)
    idx = sc.groupby(["graph_id", "hop_key", "use_reduction"], observed=True)[
        "apsp_sum"
    ].idxmin()
    return sc.loc[idx].reset_index(drop=True)


def summarize_best_of(df: pd.DataFrame) -> pd.DataFrame:
    best = best_by_config(df)
    return (
        best.groupby(GROUP, observed=True)["apsp_sum"]
        .agg(best_mean="mean", best_std="std", n_graphs_with_sc="size")
        .reset_index()
    )


def best_by_graph(df: pd.DataFrame) -> pd.DataFrame:
    sc = _sc(df)
    if sc.empty:
        return pd.DataFrame()
    idx = sc.groupby("graph_id", observed=True)["apsp_sum"].idxmin()
    columns = [
        "graph_id",
        "vertices",
        "remove_ratio_target",
        "run_id",
        "hop_key",
        "use_reduction",
        "rep",
        "apsp_sum",
        "flow_score",
        "elapsed_sec",
    ]
    if "orientation_bits" in sc:
        columns.append("orientation_bits")
    return (
        sc.loc[idx, columns]
        .sort_values(["vertices", "remove_ratio_target", "graph_id"])
        .reset_index(drop=True)
    )


def best_config_counts(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    counts = (
        best.groupby(["vertices", "hop_key", "use_reduction"], observed=True)
        .size()
        .rename("n_best")
        .reset_index()
    )
    total = best.groupby("vertices")["graph_id"].size().rename("n_graphs")
    return counts.merge(total, on="vertices")


# --- 짝지은 비교 --------------------------------------------------------------


def paired_reduction(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["graph_id", "vertices", "remove_ratio_target", "hop_key", "rep"]
    values = ["apsp_sum", "elapsed_sec", "qubo_vars_total", "strongly_connected", "ok"]
    wide = df.pivot_table(
        index=keys,
        columns="use_reduction",
        values=values,
        aggfunc="first",
        observed=True,
    )
    wide.columns = [
        f"{value}_{'on' if flag else 'off'}" for value, flag in wide.columns
    ]
    wide = wide.reset_index()
    if "ok_on" not in wide or "ok_off" not in wide:
        return pd.DataFrame()
    wide = wide[
        wide["ok_on"].fillna(False).astype(bool)
        & wide["ok_off"].fillna(False).astype(bool)
    ]
    wide["both_sc"] = wide["strongly_connected_on"].astype(bool) & wide[
        "strongly_connected_off"
    ].astype(bool)
    wide["diff_apsp"] = wide["apsp_sum_on"] - wide["apsp_sum_off"]
    wide["diff_time"] = wide["elapsed_sec_on"] - wide["elapsed_sec_off"]
    wide["ratio_time"] = wide["elapsed_sec_on"] / wide["elapsed_sec_off"]
    wide["diff_qubo_vars"] = wide["qubo_vars_total_on"] - wide["qubo_vars_total_off"]
    return wide.reset_index(drop=True)


def _wilcoxon(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    from scipy.stats import wilcoxon

    mask = a.notna() & b.notna()
    x, y = a[mask].to_numpy(float), b[mask].to_numpy(float)
    n = len(x)
    if n < 5 or np.allclose(x, y):
        return math.nan, float(np.median(x - y)) if n else math.nan, n
    return float(wilcoxon(x, y).pvalue), float(np.median(x - y)), n


def wilcoxon_reduction(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if paired.empty:
        return pd.DataFrame()
    for (vertices, hop_key), group in paired.groupby(
        ["vertices", "hop_key"], observed=True
    ):
        p_time, med_time, _ = _wilcoxon(
            group["elapsed_sec_on"], group["elapsed_sec_off"]
        )
        sc = group[group["both_sc"]]
        p_apsp, med_apsp, n_apsp = _wilcoxon(sc["apsp_sum_on"], sc["apsp_sum_off"])
        rows.append(
            {
                "vertices": vertices,
                "hop_key": hop_key,
                "n_pairs": len(group),
                "n_pairs_both_sc": n_apsp,
                "sc_rate_on": group["strongly_connected_on"].mean(),
                "sc_rate_off": group["strongly_connected_off"].mean(),
                "median_diff_apsp": med_apsp,
                "p_apsp": p_apsp,
                "median_diff_time": med_time,
                "median_ratio_time": group["ratio_time"].median(),
                "p_time": p_time,
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_hops(df: pd.DataFrame, baseline: str = "h2") -> pd.DataFrame:
    keys = ["graph_id", "rep"]
    rows = []
    ok = df[df["ok"]]
    for (vertices, use_reduction), group in ok.groupby(
        ["vertices", "use_reduction"], observed=True
    ):
        base = group[group["hop_key"] == baseline].set_index(keys)
        for hop_key in HOP_ORDER:
            if hop_key == baseline:
                continue
            other = group[group["hop_key"] == hop_key].set_index(keys)
            joined = base.join(other, lsuffix="_base", rsuffix="_other", how="inner")
            if joined.empty:
                continue
            both = joined[
                joined["strongly_connected_base"] & joined["strongly_connected_other"]
            ]
            p, med, n = _wilcoxon(both["apsp_sum_other"], both["apsp_sum_base"])
            p_t, med_t, _ = _wilcoxon(
                joined["elapsed_sec_other"], joined["elapsed_sec_base"]
            )
            rows.append(
                {
                    "vertices": vertices,
                    "use_reduction": use_reduction,
                    "hop_key": hop_key,
                    "baseline": baseline,
                    "n_pairs": len(joined),
                    "n_pairs_both_sc": n,
                    "sc_rate_hop": joined["strongly_connected_other"].mean(),
                    "sc_rate_base": joined["strongly_connected_base"].mean(),
                    "median_diff_apsp": med,
                    "p_apsp": p,
                    "median_diff_time": med_t,
                    "p_time": p_t,
                }
            )
    return pd.DataFrame(rows)


# --- 시간 추정 --------------------------------------------------------------


def estimate_total_hours(
    df: pd.DataFrame, specs: list[RunSpec], workers: int
) -> pd.DataFrame:
    """파일럿 (hop, 축약) 별 log(time) ~ a + b·log(v) 적합으로 전체 매트릭스 시간을 추정한다."""
    rows = []
    timed = df[df["elapsed_sec"].notna()]
    for (hop_key, use_reduction), group in timed.groupby(
        ["hop_key", "use_reduction"], observed=True
    ):
        per_v = group.groupby("vertices")["elapsed_sec"].mean()
        if len(per_v) >= 2:
            b, a = np.polyfit(
                np.log(per_v.index.to_numpy(float)), np.log(per_v.to_numpy(float)), 1
            )
        else:
            a, b = math.log(float(per_v.iloc[0])), 0.0
        targets = [
            s
            for s in specs
            if s.hop_key == hop_key and s.use_reduction == use_reduction
        ]
        by_v: dict[int, int] = {}
        for spec in targets:
            by_v[spec.vertices] = by_v.get(spec.vertices, 0) + 1
        for vertices, count in sorted(by_v.items()):
            predicted = math.exp(a + b * math.log(vertices))
            rows.append(
                {
                    "hop_key": hop_key,
                    "use_reduction": use_reduction,
                    "vertices": vertices,
                    "n_runs": count,
                    "pred_sec_per_run": predicted,
                    "measured_sec_per_run": float(per_v.get(vertices, math.nan)),
                    "exponent_b": b,
                    "total_hours_sequential": predicted * count / 3600,
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["total_hours_parallel"] = table["total_hours_sequential"] / workers
    return table


# --- 해 저장·검증 ----------------------------------------------------------------


def write_solutions_jsonl(df: pd.DataFrame, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "orientation_bits" not in df:
        return 0
    written = 0
    columns = [
        "run_id",
        "hop_key",
        "use_reduction",
        "rep",
        "run_seed",
        "status",
        "strongly_connected",
        "apsp_sum",
        "flow_score",
        "orientation_bits",
    ]
    for graph_id, group in df.groupby("graph_id", observed=True):
        with (out_dir / f"{graph_id}.jsonl").open("w") as handle:
            for _, row in group.sort_values("run_id").iterrows():
                record = {c: row[c] for c in columns if c in row}
                record = {
                    k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in record.items()
                }
                record["hop_key"] = str(record["hop_key"])
                record["use_reduction"] = bool(record["use_reduction"])
                handle.write(json.dumps(record, default=_json_default) + "\n")
                written += 1
    return written


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"unserializable {type(value)}")


def verify_solutions(
    df: pd.DataFrame, graph_dir: Path, tolerance: float = 1e-9
) -> pd.DataFrame:
    """저장된 비트열을 복원·재평가해 기록된 점수와 대조한다. 불일치 행만 돌려준다."""
    mismatches = []
    ok = df[df["ok"]]
    records = {}
    for _, row in ok.iterrows():
        record = records.get(row["graph_id"])
        if record is None:
            record = records[row["graph_id"]] = load_graph(
                graph_path(graph_dir, row["graph_id"])
            )
        score = reevaluate(load_solution(record, row["orientation_bits"]))
        stored_apsp = row["apsp_sum"] if not pd.isna(row["apsp_sum"]) else math.inf
        apsp_ok = math.isclose(
            score.apsp_sum, stored_apsp, rel_tol=tolerance, abs_tol=tolerance
        ) or (math.isinf(score.apsp_sum) and math.isinf(stored_apsp))
        flow_ok = math.isclose(
            score.flow_score, row["flow_score"], rel_tol=tolerance, abs_tol=tolerance
        )
        if not (apsp_ok and flow_ok):
            mismatches.append(
                {
                    "run_id": row["run_id"],
                    "stored_apsp": stored_apsp,
                    "restored_apsp": score.apsp_sum,
                    "stored_flow": row["flow_score"],
                    "restored_flow": score.flow_score,
                }
            )
    return pd.DataFrame(mismatches)


# --- 그림 -------------------------------------------------------------------


def _style(ax: Any) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def make_figures(
    df: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame, out_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # 한글 축 라벨용. 앞에서부터 설치된 글꼴을 쓰고 없으면 DejaVu 로 떨어진다.
            "font.family": [
                "Apple SD Gothic Neo",
                "AppleGothic",
                "NanumGothic",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "axes.titlesize": 10,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = sorted(summary["remove_ratio_target"].unique())
    written: list[Path] = []

    # 1. hop 별 stretch vs v (축약 on), 제거 비율별 패널
    fig, axes = plt.subplots(
        1, len(ratios), figsize=(3.4 * len(ratios), 3.2), sharey=True
    )
    for ax, ratio in zip(np.atleast_1d(axes), ratios, strict=True):
        part = summary[
            (summary["remove_ratio_target"] == ratio) & summary["use_reduction"]
        ]
        for hop_key in HOP_ORDER:
            line = part[part["hop_key"] == hop_key].sort_values("vertices")
            if line.empty or line["apsp_mean"].isna().all():
                continue
            ax.errorbar(
                line["vertices"],
                line["apsp_mean"],
                yerr=line["apsp_std"].fillna(0),
                color=HOP_COLORS[hop_key],
                marker="o",
                capsize=2,
                label=hop_key,
            )
        ax.set_title(f"간선 제거 {int(ratio * 100)}%")
        ax.set_xlabel("정점 수 v")
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("평균 stretch (강연결 해, 축약 on)")
    np.atleast_1d(axes)[0].legend(frameon=False, title="hop")
    fig.tight_layout()
    written.append(out_dir / "fig_hop_stretch.pdf")
    fig.savefig(written[-1])
    plt.close(fig)

    # 2. hop 별 강연결 성공률 (축약 on/off 패널)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, use_reduction in zip(axes, (True, False), strict=True):
        part = summary[summary["use_reduction"] == use_reduction]
        vs = sorted(part["vertices"].unique())
        width = 0.8 / len(HOP_ORDER)
        for i, hop_key in enumerate(HOP_ORDER):
            line = (
                part[part["hop_key"] == hop_key].groupby("vertices")["sc_rate"].mean()
            )
            ax.bar(
                [vs.index(v) + i * width for v in line.index],
                line.values,
                width=width * 0.9,
                color=HOP_COLORS[hop_key],
                label=hop_key,
            )
        ax.set_xticks(
            [i + 0.4 - width / 2 for i in range(len(vs))], [str(v) for v in vs]
        )
        ax.set_title(REDUCTION_LABEL[use_reduction])
        ax.set_xlabel("정점 수 v")
        ax.set_ylim(0, 1.05)
        _style(ax)
    axes[0].set_ylabel("강연결 성공률")
    axes[0].legend(frameon=False, title="hop", fontsize=8)
    fig.tight_layout()
    written.append(out_dir / "fig_hop_sc_rate.pdf")
    fig.savefig(written[-1])
    plt.close(fig)

    # 3. 실행 시간 vs v (실선 on, 점선 off)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for hop_key in HOP_ORDER:
        for use_reduction in (True, False):
            line = summary[
                (summary["hop_key"] == hop_key)
                & (summary["use_reduction"] == use_reduction)
            ]
            line = line.groupby("vertices")["time_mean"].mean()
            if line.empty:
                continue
            ax.plot(
                line.index,
                line.values,
                REDUCTION_STYLE[use_reduction],
                color=HOP_COLORS[hop_key],
                marker="o",
                label=f"{hop_key} {REDUCTION_LABEL[use_reduction]}",
            )
    ax.set_yscale("log")
    ax.set_xlabel("정점 수 v")
    ax.set_ylabel("실행 시간 (초, 로그)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _style(ax)
    fig.tight_layout()
    written.append(out_dir / "fig_reduction_time.pdf")
    fig.savefig(written[-1])
    plt.close(fig)

    # 4. 축약 on − off 짝지은 stretch 차이 (hop 별 패널, v 별 박스)
    if not paired.empty:
        fig, axes = plt.subplots(
            1, len(HOP_ORDER), figsize=(2.6 * len(HOP_ORDER), 3.2), sharey=True
        )
        for ax, hop_key in zip(axes, HOP_ORDER, strict=True):
            part = paired[(paired["hop_key"] == hop_key) & paired["both_sc"]]
            vs = sorted(part["vertices"].unique())
            data = [
                part[part["vertices"] == v]["diff_apsp"].dropna().to_numpy() for v in vs
            ]
            if vs:
                ax.boxplot(
                    data,
                    tick_labels=[str(v) for v in vs],
                    widths=0.5,
                    boxprops={"color": HOP_COLORS[hop_key]},
                    medianprops={"color": "#0b0b0b"},
                )
            ax.axhline(0, color="#52514e", linewidth=0.8)
            ax.set_title(hop_key)
            ax.set_xlabel("정점 수 v")
            _style(ax)
        axes[0].set_ylabel("stretch 차이 (축약 on − off)")
        fig.tight_layout()
        written.append(out_dir / "fig_reduction_paired_stretch.pdf")
        fig.savefig(written[-1])
        plt.close(fig)

    # 5. QUBO 변수 수 vs v
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for hop_key in HOP_ORDER:
        for use_reduction in (True, False):
            line = summary[
                (summary["hop_key"] == hop_key)
                & (summary["use_reduction"] == use_reduction)
            ]
            line = line.groupby("vertices")["qubo_vars_mean"].mean().dropna()
            if line.empty:
                continue
            ax.plot(
                line.index,
                line.values,
                REDUCTION_STYLE[use_reduction],
                color=HOP_COLORS[hop_key],
                marker="o",
                label=f"{hop_key} {REDUCTION_LABEL[use_reduction]}",
            )
    ax.set_yscale("log")
    ax.set_xlabel("정점 수 v")
    ax.set_ylabel("QUBO 변수 수 합계 (로그)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _style(ax)
    fig.tight_layout()
    written.append(out_dir / "fig_reduction_qubo_vars.pdf")
    fig.savefig(written[-1])
    plt.close(fig)
    return written


# --- 마크다운 요약 -------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pivot_md(summary: pd.DataFrame, use_reduction: bool, cell: Any, title: str) -> str:
    part = summary[summary["use_reduction"] == use_reduction]
    lines = [
        f"### {title} ({REDUCTION_LABEL[use_reduction]})",
        "",
        "| v | 제거 | " + " | ".join(HOP_ORDER) + " |",
        "|---|---|" + "---|" * len(HOP_ORDER),
    ]
    for (vertices, ratio), group in part.groupby(["vertices", "remove_ratio_target"]):
        cells = []
        for hop_key in HOP_ORDER:
            row = group[group["hop_key"] == hop_key]
            cells.append(cell(row.iloc[0]) if not row.empty else "–")
        lines.append(
            f"| {vertices} | {int(ratio * 100)}% | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"


def write_summary_md(
    summary: pd.DataFrame,
    best_of: pd.DataFrame,
    wil_red: pd.DataFrame,
    wil_hops: pd.DataFrame,
    counts: pd.DataFrame,
    path: Path,
) -> None:
    merged = summary.merge(best_of, on=GROUP, how="left")
    parts = ["# 실험 요약 (자동 생성)", ""]
    for use_reduction in (True, False):
        parts.append(
            _pivot_md(
                merged,
                use_reduction,
                lambda r: (
                    f"{_fmt(r['apsp_mean'])} ± {_fmt(r['apsp_std'])} (n={int(r['n_sc'])}/{int(r['n_runs'])})"
                ),
                "평균 stretch ± std (강연결 해만, n=강연결/전체)",
            )
        )
        parts.append(
            _pivot_md(
                merged,
                use_reduction,
                lambda r: _fmt(r["best_mean"]),
                "best-of-반복 stretch 평균 (그래프별 최선의 평균)",
            )
        )
        parts.append(
            _pivot_md(
                merged,
                use_reduction,
                lambda r: (
                    f"{_fmt(r['sc_rate'], 2)} (t/o {int(r['n_timeout'])}, err {int(r['n_error'])})"
                ),
                "강연결 성공률",
            )
        )
        parts.append(
            _pivot_md(
                merged,
                use_reduction,
                lambda r: f"{_fmt(r['time_mean'], 1)} ± {_fmt(r['time_std'], 1)}",
                "실행 시간 (초)",
            )
        )
        parts.append(
            _pivot_md(
                merged,
                use_reduction,
                lambda r: (
                    f"{_fmt(r['qubo_vars_mean'], 0)} / sub {_fmt(r['n_subgraphs_mean'], 1)}"
                ),
                "QUBO 변수 수 합계 / DnC 서브그래프 수",
            )
        )
    if not wil_red.empty:
        parts.append("### 축약 on vs off (짝지은 Wilcoxon, 같은 그래프·같은 반복)\n")
        parts.append(wil_red.to_markdown(index=False, floatfmt=".4g"))
        parts.append("")
    if not wil_hops.empty:
        parts.append("### hop 조합 vs h2 (짝지은 Wilcoxon)\n")
        parts.append(wil_hops.to_markdown(index=False, floatfmt=".4g"))
        parts.append("")
    if not counts.empty:
        parts.append("### 그래프별 최선 해를 낸 구성의 빈도\n")
        parts.append(counts.to_markdown(index=False))
        parts.append("")
    path.write_text("\n".join(parts))


# --- CLI ---------------------------------------------------------------------


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MR2S 실험 결과 집계")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument(
        "--estimate", action="store_true", help="전체 매트릭스 예상 시간 산출"
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--hop4-max-vertices", type=int, default=config.HOP4_MAX_VERTICES
    )
    parser.add_argument(
        "--verify", action="store_true", help="모든 해를 복원·재평가해 대조"
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    results_dir: Path = args.results
    df = collect_runs(results_dir / "runs")
    if df.empty:
        print("no runs found")
        return
    write_results_csv(df, results_dir / "results.csv")
    summary = summarize_by_config(df)
    summary.to_csv(results_dir / "summary_by_config.csv", index=False)
    best_of = summarize_best_of(df)
    best_of.to_csv(results_dir / "best_of_by_config.csv", index=False)
    paired = paired_reduction(df)
    paired.to_csv(results_dir / "paired_reduction.csv", index=False)
    wil_red = wilcoxon_reduction(paired)
    wil_red.to_csv(results_dir / "wilcoxon_reduction.csv", index=False)
    wil_hops = wilcoxon_hops(df)
    wil_hops.to_csv(results_dir / "wilcoxon_hops.csv", index=False)
    best = best_by_graph(df)
    best.to_csv(results_dir / "best_by_graph.csv", index=False)
    counts = best_config_counts(best)
    counts.to_csv(results_dir / "best_config_counts.csv", index=False)
    n_solutions = write_solutions_jsonl(df, results_dir / "solutions")
    write_summary_md(
        summary, best_of, wil_red, wil_hops, counts, results_dir / "summary.md"
    )
    print(
        f"runs={len(df)} ok={int(df['ok'].sum())} timeout={int((df['status'] == 'timeout').sum())} "
        f"error={int((df['status'] == 'error').sum())} solutions={n_solutions}"
    )
    if not args.no_figures:
        for path in make_figures(df, summary, paired, results_dir / "figures"):
            print(f"figure: {path}")
    if args.estimate:
        specs = iter_run_specs(hop4_max_vertices=args.hop4_max_vertices)
        table = estimate_total_hours(df, specs, args.workers)
        table.to_csv(results_dir / "estimate.csv", index=False)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(
            f"TOTAL ≈ {table['total_hours_parallel'].sum():.1f} h with {args.workers} workers"
        )
    if args.verify:
        mismatches = verify_solutions(df, args.graph_dir)
        print(f"verify: {len(mismatches)} mismatches")
        if not mismatches.empty:
            print(mismatches.to_string(index=False))


if __name__ == "__main__":
    main()
