import pytest

from experiments import config
from experiments.config import RunSpec, iter_run_specs, parse_run_id


def test_run_id_roundtrip() -> None:
    spec = RunSpec(200, 3, 0.3, "h2+3", True, 4)
    assert spec.run_id == "v200_s3_p30__h2+3__red__r4"
    assert parse_run_id(spec.run_id) == spec
    nored = RunSpec(100, 0, 0.0, "h4", False, 0)
    assert parse_run_id(nored.run_id) == nored


def test_run_seed_shared_across_configs_of_same_graph_and_rep() -> None:
    a = RunSpec(100, 0, 0.1, "h2", True, 1)
    b = RunSpec(100, 0, 0.1, "h2+3+4", False, 1)
    c = RunSpec(100, 0, 0.1, "h2", True, 2)
    assert a.run_seed == b.run_seed
    assert a.run_seed != c.run_seed
    assert 0 <= a.run_seed < 2**31


def test_matrix_size_and_order() -> None:
    specs = iter_run_specs(
        vertex_counts=(100, 200),
        graph_seeds=(0, 1),
        remove_ratios=(0.0, 0.5),
        reps=2,
        hop4_max_vertices=None,
    )
    assert len(specs) == 2 * 2 * 2 * len(config.HOP_SETS) * 2 * 2
    assert len({spec.run_id for spec in specs}) == len(specs)
    half = len(specs) // 2
    assert all(spec.rep == 0 for spec in specs[:half])
    assert all(spec.rep == 1 for spec in specs[half:])


def test_hop4_max_vertices_filters_hop4_configs() -> None:
    specs = iter_run_specs(
        vertex_counts=(100, 300),
        graph_seeds=(0,),
        remove_ratios=(0.0,),
        reps=1,
        hop4_max_vertices=200,
    )
    big_hop4 = [s for s in specs if s.vertices == 300 and 4 in s.hops]
    small_hop4 = [s for s in specs if s.vertices == 100 and 4 in s.hops]
    assert not big_hop4
    assert len(small_hop4) == 2 * 2  # h4, h2+3+4 × 축약 on/off


def test_whole_graph_run_id_roundtrip_and_matrix() -> None:
    whole = RunSpec(100, 0, 0.3, "h2", True, 0, use_dnc=False)
    assert whole.run_id == "v100_s0_p30__h2__red__whole__r0"
    assert parse_run_id(whole.run_id) == whole
    assert parse_run_id("v100_s0_p30__h2__red__r0").use_dnc is True
    with pytest.raises(ValueError):
        parse_run_id("v100_s0_p30__h2__red__dnc__r0")
    specs = iter_run_specs(
        vertex_counts=(100,),
        graph_seeds=(0,),
        remove_ratios=(0.0,),
        reps=1,
        dnc_modes=(False,),
    )
    assert len(specs) == len(config.HOP_SETS) * 2 and all(not s.use_dnc for s in specs)
    assert all(
        s.run_seed == RunSpec(100, 0, 0.0, "h2", True, 0).run_seed for s in specs
    )
