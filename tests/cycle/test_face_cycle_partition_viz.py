"""
파티션 시각화 테스트.

`FaceCycleProtocol` 구현체가 반환한 `GraphPartitionResult` 를 같은 렌더러로
그려서 구현체별 partition PNG 를 저장한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")
pytest.importorskip("scipy.spatial")

from mr2s_module.cycle import (
    BalancedFaceGraphClusterer,
    FaceCycle,
    KMeansFaceClusterer,
    PlanarRegionFaceCycle,
    SnowballFaceClusterer,
)

from tests.cycle.partition_visualization import (
    delaunay_graph_with_pos,
    partition_balance_report,
    render_face_cycle_partition_png,
)

_OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.mark.parametrize("seed,n_points,target_k", [(42, 60, 6), (7, 80, 8)])
@pytest.mark.parametrize(
    "name,face_cycle_factory",
    [
        (
            "face_cycle_snowball",
            lambda target_k: FaceCycle(
                target_k=target_k,
                clusterer=SnowballFaceClusterer(),
            ),
        ),
        (
            "face_cycle_kmeans",
            lambda target_k: FaceCycle(
                target_k=target_k,
                clusterer=KMeansFaceClusterer(),
            ),
        ),
        (
            "face_cycle_balanced",
            lambda target_k: FaceCycle(
                target_k=target_k,
                clusterer=BalancedFaceGraphClusterer(),
            ),
        ),
        ("planar_region_face_cycle", PlanarRegionFaceCycle),
    ],
)
def test_face_cycle_partition_visualization_renders(
    seed: int,
    n_points: int,
    target_k: int,
    name,
    face_cycle_factory,
) -> None:
    graph, pos = delaunay_graph_with_pos(n=n_points, seed=seed)
    face_cycle = face_cycle_factory(target_k=target_k)
    np.random.seed(seed)

    out_path = (
        _OUTPUT_DIR
        / f"{name}_partition_seed{seed}_n{n_points}_k{target_k}.png"
    )
    partition = render_face_cycle_partition_png(
        graph=graph,
        pos=pos,
        face_cycle=face_cycle,
        path=out_path,
        title=(
            f"{name} partition - n={n_points}, target_k={target_k}, "
            f"seed={seed}"
        ),
    )

    assert len(partition.sub_graphs) > 0
    balance = partition_balance_report(partition)
    print(
        f"{name} seed={seed} n={n_points} k={target_k} "
        f"sizes={balance['sizes']} "
        f"target={balance['target']:.2f} "
        f"max_deviation={balance['max_deviation']:.2f} "
        f"mean_deviation={balance['mean_deviation']:.2f} "
        f"balance_score={balance['score']:.1f}/100"
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0
