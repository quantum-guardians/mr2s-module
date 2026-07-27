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
    FaceClusterPartition,
    KMeansFaceClusterer,
    SnowballFaceClusterer,
)
from mr2s_module.util import inner_faces_by_edge_id
from tests.cycle.partition_visualization import (
    draw_partition,
    partition_balance_report,
    render_face_cycle_partition_png,
)
from tests.util.graph_fixtures import delaunay_graph_with_pos

_OUTPUT_DIR = Path(__file__).parent / "output"


def test_draw_partition_fills_every_inner_face() -> None:
    """면 색칠이 실제로 폴리곤을 그리는지 검증한다.

    이전 구현은 정점쌍 키(face_edges)와 edge id 집합을 교집합해서 owner 가 항상
    None 이었고, PNG 존재만 보는 테스트라 색칠이 100% 죽은 걸 못 잡았다.
    """
    import matplotlib.pyplot as plt

    graph, pos = delaunay_graph_with_pos(n=60, seed=42)
    face_cycle = FaceClusterPartition(target_k=6, clusterer=KMeansFaceClusterer())
    np.random.seed(42)
    partition = face_cycle.run(graph)

    fig, ax = plt.subplots()
    try:
        draw_partition(ax, graph, partition, pos)
        assert len(ax.patches) == len(inner_faces_by_edge_id(graph, pos))
        assert len(ax.patches) > 0
    finally:
        plt.close(fig)


@pytest.mark.parametrize("seed,n_points,target_k", [(42, 60, 6), (7, 80, 8)])
@pytest.mark.parametrize(
    "name,face_cycle_factory",
    [
        (
            "face_cycle_snowball",
            lambda target_k: FaceClusterPartition(
                target_k=target_k,
                clusterer=SnowballFaceClusterer(),
            ),
        ),
        (
            "face_cycle_kmeans",
            lambda target_k: FaceClusterPartition(
                target_k=target_k,
                clusterer=KMeansFaceClusterer(),
            ),
        ),
        (
            "face_cycle_balanced",
            lambda target_k: FaceClusterPartition(
                target_k=target_k,
                clusterer=BalancedFaceGraphClusterer(),
            ),
        ),
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

    out_path = _OUTPUT_DIR / f"{name}_partition_seed{seed}_n{n_points}_k{target_k}.png"
    partition = render_face_cycle_partition_png(
        graph=graph,
        pos=pos,
        face_cycle=face_cycle,
        path=out_path,
        title=(f"{name} partition - n={n_points}, target_k={target_k}, seed={seed}"),
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
