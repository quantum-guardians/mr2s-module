"""실험 매트릭스 상수와 실행 단위(RunSpec) 정의.

run_id 는 파일명이자 재시작 키다. run_seed 는 (graph_id, rep) 에만 의존하므로
같은 그래프·같은 반복의 hop/축약 구성들이 동일한 seed 를 공유한다 (짝 비교용).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

VERTEX_COUNTS: tuple[int, ...] = (100, 200, 300, 400, 500)
GRAPH_SEEDS: tuple[int, ...] = tuple(range(10))
REMOVE_RATIOS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5)
HOP_SETS: dict[str, tuple[int, ...]] = {
    "h2": (2,),
    "h3": (3,),
    "h4": (4,),
    "h2+3": (2, 3),
    "h2+3+4": (2, 3, 4),
}
REDUCTION_MODES: tuple[bool, ...] = (True, False)
REPS = 5
NUM_READS = 100
# 4-hop 계열(h4, h2+3+4)을 실행할 최대 정점 수. None 이면 제한 없음. 파일럿 후 확정.
HOP4_MAX_VERTICES: int | None = None
# 실행 1회의 벽시계 타임아웃(초). 파일럿 후 확정.
TIMEOUT_SEC_BY_VERTICES: dict[int, int] = {
    100: 600,
    200: 1200,
    300: 1800,
    400: 2400,
    500: 3600,
}

_GRAPH_ID_RE = re.compile(r"^v(?P<v>\d+)_s(?P<s>\d+)_p(?P<p>\d+)$")


def graph_id(vertices: int, seed: int, remove_ratio: float) -> str:
    return f"v{vertices}_s{seed}_p{round(remove_ratio * 100)}"


def parse_graph_id(value: str) -> tuple[int, int, float]:
    """graph_id → (vertices, seed, remove_ratio)."""
    match = _GRAPH_ID_RE.match(value)
    if match is None:
        raise ValueError(f"invalid graph_id: {value!r}")
    return int(match["v"]), int(match["s"]), int(match["p"]) / 100


def hops_of(hop_key: str) -> tuple[int, ...]:
    try:
        return HOP_SETS[hop_key]
    except KeyError as exc:
        raise ValueError(f"unknown hop_key: {hop_key!r}") from exc


def uses_hop4(hop_key: str) -> bool:
    return 4 in hops_of(hop_key)


@dataclass(frozen=True)
class RunSpec:
    vertices: int
    graph_seed: int
    remove_ratio: float
    hop_key: str
    use_reduction: bool
    rep: int
    use_dnc: bool = True  # False = DnC 없이 그래프 전체를 한 QUBO 로 푸는 대조군

    @property
    def graph_id(self) -> str:
        return graph_id(self.vertices, self.graph_seed, self.remove_ratio)

    @property
    def hops(self) -> tuple[int, ...]:
        return hops_of(self.hop_key)

    @property
    def reduction_tag(self) -> str:
        return "red" if self.use_reduction else "nored"

    @property
    def run_id(self) -> str:
        # DnC 사용(기본)은 태그 없이 4부분, 전체 그래프 대조군은 "whole" 태그를 끼운 5부분.
        dnc_tag = "" if self.use_dnc else "__whole"
        return f"{self.graph_id}__{self.hop_key}__{self.reduction_tag}{dnc_tag}__r{self.rep}"

    @property
    def run_seed(self) -> int:
        digest = hashlib.sha256(f"{self.graph_id}__r{self.rep}".encode()).digest()
        # 31비트: dwave.samplers SA 와 numpy 의 seed 상한을 모두 만족한다.
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def parse_run_id(run_id: str) -> RunSpec:
    parts = run_id.split("__")
    use_dnc = True
    if len(parts) == 5:
        if parts[3] != "whole":
            raise ValueError(f"invalid run_id: {run_id!r}")
        use_dnc = False
        parts = parts[:3] + parts[4:]
    if len(parts) != 4:
        raise ValueError(f"invalid run_id: {run_id!r}")
    gid, hop_key, reduction_tag, rep_part = parts
    vertices, seed, remove_ratio = parse_graph_id(gid)
    hops_of(hop_key)
    if reduction_tag not in ("red", "nored"):
        raise ValueError(f"invalid reduction tag in run_id: {run_id!r}")
    if not rep_part.startswith("r") or not rep_part[1:].isdigit():
        raise ValueError(f"invalid rep in run_id: {run_id!r}")
    return RunSpec(
        vertices=vertices,
        graph_seed=seed,
        remove_ratio=remove_ratio,
        hop_key=hop_key,
        use_reduction=reduction_tag == "red",
        rep=int(rep_part[1:]),
        use_dnc=use_dnc,
    )


def iter_run_specs(
    *,
    vertex_counts: Iterable[int] = VERTEX_COUNTS,
    graph_seeds: Iterable[int] = GRAPH_SEEDS,
    remove_ratios: Iterable[float] = REMOVE_RATIOS,
    hop_keys: Sequence[str] = tuple(HOP_SETS),
    reduction_modes: Iterable[bool] = REDUCTION_MODES,
    reps: int = REPS,
    hop4_max_vertices: int | None = HOP4_MAX_VERTICES,
    dnc_modes: Iterable[bool] = (True,),
) -> list[RunSpec]:
    """매트릭스 전개. rep 가 가장 바깥 루프라 중간에 멈춰도 구성별 반복 수가 균형을 이룬다."""
    vertex_list = list(vertex_counts)
    seed_list = list(graph_seeds)
    ratio_list = list(remove_ratios)
    reduction_list = list(reduction_modes)
    dnc_list = list(dnc_modes)
    for hop_key in hop_keys:
        hops_of(hop_key)

    specs: list[RunSpec] = []
    for rep in range(reps):
        for vertices in vertex_list:
            for seed in seed_list:
                for ratio in ratio_list:
                    for hop_key in hop_keys:
                        if (
                            hop4_max_vertices is not None
                            and uses_hop4(hop_key)
                            and vertices > hop4_max_vertices
                        ):
                            continue
                        for use_reduction in reduction_list:
                            for use_dnc in dnc_list:
                                specs.append(
                                    RunSpec(
                                        vertices=vertices,
                                        graph_seed=seed,
                                        remove_ratio=ratio,
                                        hop_key=hop_key,
                                        use_reduction=use_reduction,
                                        rep=rep,
                                        use_dnc=use_dnc,
                                    )
                                )
    return specs
