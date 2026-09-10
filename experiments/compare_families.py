"""그래프 계열별 파일럿 결과 비교 (summary_by_config.csv 를 계열 축으로 묶는다).

사용: python -m experiments.compare_families --results experiments/results \
          --families delaunay:pilot,grid:pilot_grid,hexagonal:pilot_hexagonal,apollonian:pilot_apollonian
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments import config

HOPS = list(config.HOP_SETS)


def load_families(results_root: Path, families: dict[str, str]) -> pd.DataFrame:
    frames = []
    for family, name in families.items():
        frame = pd.read_csv(results_root / name / "summary_by_config.csv")
        frame.insert(0, "family", family)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _fmt_ratio(value: float) -> str:
    return f"{round(value * 100)}%"


def table_by_hop(df: pd.DataFrame, column: str, fmt, *, use_reduction: bool) -> str:
    """행 = (계열, v, 제거), 열 = hop 집합."""
    sub = df[df["use_reduction"] == use_reduction]
    lines = [
        "| 계열 | v | 제거 | " + " | ".join(HOPS) + " |",
        "|---" * (3 + len(HOPS)) + "|",
    ]
    keys = ["family", "vertices", "remove_ratio_target"]
    for (family, vertices, ratio), group in sub.groupby(keys, sort=False):
        by_hop = group.set_index("hop_key")
        cells = [fmt(by_hop.loc[hop]) if hop in by_hop.index else "-" for hop in HOPS]
        lines.append(
            f"| {family} | {vertices} | {_fmt_ratio(ratio)} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def _stretch(row: pd.Series) -> str:
    if row["n_sc"] == 0:
        return f"- (n=0/{row['n_runs']})"
    return f"{row['apsp_mean']:.3f} (n={row['n_sc']}/{row['n_runs']})"


def _errors(row: pd.Series) -> str:
    return f"{int(row['n_error'])}/{int(row['n_runs'])}"


def _time(row: pd.Series) -> str:
    return f"{row['time_mean']:.1f}"


def _qubo(row: pd.Series) -> str:
    return f"{row['qubo_vars_mean']:.0f} / sub {row['n_subgraphs_mean']:.1f}"


def best_hop_by_family(df: pd.DataFrame, *, use_reduction: bool) -> str:
    """계열별로 (v, 제거) 조합마다 평균 stretch 가 가장 낮은 hop 집합을 센다."""
    sub = df[(df["use_reduction"] == use_reduction) & (df["n_sc"] > 0)]
    idx = sub.groupby(["family", "vertices", "remove_ratio_target"], sort=False)[
        "apsp_mean"
    ].idxmin()
    counts = (
        sub.loc[idx]
        .groupby(["family", "hop_key"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(columns=[h for h in HOPS if h in counts.columns])
    lines = [
        "| 계열 | " + " | ".join(counts.columns) + " |",
        "|---" * (1 + len(counts.columns)) + "|",
    ]
    for family, row in counts.iterrows():
        lines.append(f"| {family} | " + " | ".join(str(int(v)) for v in row) + " |")
    return "\n".join(lines)


def render(df: pd.DataFrame) -> str:
    sections = ["# 그래프 계열별 파일럿 비교 (자동 생성)", ""]
    for use_reduction in (True, False):
        tag = "축약 on" if use_reduction else "축약 off"
        sections += [
            f"## {tag}",
            "",
            "### 평균 stretch (강연결 해만, n=강연결/전체)",
            "",
            table_by_hop(df, "apsp_mean", _stretch, use_reduction=use_reduction),
            "",
            "### (v, 제거) 조합별 최저 stretch hop 집합 횟수",
            "",
            best_hop_by_family(df, use_reduction=use_reduction),
            "",
            "### DnC 분할 실패 (error/전체)",
            "",
            table_by_hop(df, "n_error", _errors, use_reduction=use_reduction),
            "",
            "### 실행 시간 평균 (초)",
            "",
            table_by_hop(df, "time_mean", _time, use_reduction=use_reduction),
            "",
            "### QUBO 변수 수 합계 / DnC 서브그래프 수",
            "",
            table_by_hop(df, "qubo_vars_mean", _qubo, use_reduction=use_reduction),
            "",
        ]
    return "\n".join(sections)


def _parse_families(text: str) -> dict[str, str]:
    pairs = (part.split(":", 1) for part in text.split(",") if part)
    return dict(pairs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="그래프 계열별 파일럿 비교")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--families",
        type=_parse_families,
        default=_parse_families(
            "delaunay:pilot,grid:pilot_grid,hexagonal:pilot_hexagonal,apollonian:pilot_apollonian"
        ),
        help="family:results_subdir 쉼표 목록",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = render(load_families(args.results, args.families))
    out = args.out or args.results / "families_pilot.md"
    out.write_text(text + "\n")
    print(out)


if __name__ == "__main__":
    main()
